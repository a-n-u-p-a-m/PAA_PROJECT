"""Correctness tests for the CLEAR capture / replay / compose path.

An energy number from CLEAR is only as good as the replay behind it: if the
replayed call is not the call decoding made, the joules describe something
else, and nothing downstream will notice. These tests pin down the ways that
goes wrong before any GPU time is spent:

  * hooks see the shapes the engine really uses (draft q=1 or 2, verify K+1)
  * every replay reproduces its captured output exactly
  * replaying attention does not grow the KV cache
  * the call counts reconstructed from round_detail match the real calls
  * hooks do not change what decoding emits

Runs on CPU with small randomly-initialised models -- no downloads, no GPU.

    python -m pytest tests/ -q     (or just: python tests/test_clear.py)
"""

from __future__ import annotations

import copy
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specdec import clear  # noqa: E402
from specdec.engine import autoregressive_generate, speculative_generate  # noqa: E402


def _tiny_model(seed: int, vocab: int = 128, layers: int = 3):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=vocab,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    return Qwen2ForCausalLM(cfg).eval()


def _perturbed(model, scale: float = 0.05, seed: int = 99):
    torch.manual_seed(seed)
    clone = copy.deepcopy(model)
    with torch.no_grad():
        for p in clone.parameters():
            p.add_(torch.randn_like(p) * scale * p.std().clamp(min=1e-6))
    return clone.eval()


def _prompt(vocab: int = 128, length: int = 8, seed: int = 7):
    torch.manual_seed(seed)
    return torch.randint(0, vocab, (1, length))


def _pair():
    target = _tiny_model(seed=0)
    return _perturbed(target, scale=0.3), target


def _store(draft, target, **kw) -> clear.ActivationStore:
    store = clear.ActivationStore(kv_bin=8, **kw)
    store.attach(draft, "draft", clear.representative_layers(3))
    store.attach(target, "target", clear.representative_layers(3))
    return store


def test_capture_sees_engine_shapes():
    """Draft calls are 1 row, or 2 after a fully accepted round; verify is K+1."""
    draft, target = _pair()
    k = 4
    with _store(draft, target) as store:
        out = speculative_generate(draft, target, _prompt(), k=k, max_new_tokens=32)

    fwd = [key for key, _ in store.items() if key.component == "forward"]
    draft_rows = {key.rows for key in fwd if key.role == "draft"}
    target_rows = {key.rows for key in fwd if key.role == "target"}
    assert draft_rows <= {1, 2}, f"draft forward rows {draft_rows}"
    assert target_rows == {k + 1}, f"verify forward rows {target_rows}"

    full = any(r.accepted == r.proposed for r in out.rounds[:-1])
    assert (2 in draft_rows) == full, (
        "a 2-row draft call should appear exactly when some round accepted all drafts"
    )
    lm = {key.rows for key, _ in store.items()
          if key.component == "lm_head" and key.role == "draft"}
    assert lm == {1}, f"draft lm_head rows {lm}; logits_to_keep should keep 1"

    prefill = [key for key, _ in store.items() if key.component == "prefill"]
    assert {key.role for key in prefill} == {"draft", "target"}
    print(f"  ok  draft rows {sorted(draft_rows)}, verify rows {sorted(target_rows)}, "
          f"prefill captured for both roles, {len(store.items())} keys")


def test_replay_reproduces_output():
    """Every captured call, replayed once, gives back exactly what it gave."""
    draft, target = _pair()
    with _store(draft, target) as store:
        speculative_generate(draft, target, _prompt(), k=3, max_new_tokens=32)
        autoregressive_generate(target, _prompt(), max_new_tokens=16)
    report = clear.check_replay(store, "cpu")
    bad = [r for r in report if not (r["exact"] and r["cache_restored"])]
    assert not bad, f"{len(bad)} replays diverged, e.g. {bad[:3]}"
    comps = sorted({r["key"].split("/")[1] for r in report})
    print(f"  ok  {len(report)} replays bit-exact across {comps}")


def test_replay_does_not_grow_cache():
    """The trap: 10,000 replays of attention against a live cache would see
    10,000 extra positions. Repeats must cost the same and return the same."""
    draft, target = _pair()
    with _store(draft, target) as store:
        speculative_generate(draft, target, _prompt(), k=3, max_new_tokens=32)

    attn = next(c for key, caps in store.items() if key.component == "attn"
                and key.role == "target" for c in caps)
    fn = clear.replay_fn(store, attn, "cpu")
    first = clear._output_tensors(fn())[0]
    for _ in range(50):
        fn()
    last = clear._output_tensors(fn())[0]
    assert torch.equal(first, last), "attention output drifted across repeats"

    fwd = next(c for key, caps in store.items() if key.component == "forward"
               and key.role == "target" for c in caps)
    fn = clear.replay_fn(store, fwd, "cpu")
    for _ in range(20):
        fn()
    assert clear.cache_length(fn.cache) == fwd.kv, (
        f"cache length {clear.cache_length(fn.cache)} after 20 replays, want {fwd.kv}"
    )

    # and the naive way really does grow, so the test is not vacuous. One
    # call is enough to show it: a second would either crash on the mask, or,
    # where the mask is None (single-row decode), silently attend to more.
    live = clear.new_cache()
    k, v = attn.kv_state
    for i in range(attn.layer + 1):
        live.update(k.clone(), v.clone(), i)
    module = store.modules[("target", "attn", attn.layer)]
    module(*attn.args, **{**attn.kwargs, "past_key_values": live})
    grown = clear.layer_kv(live, attn.layer)[0].shape[-2]
    assert grown == attn.kv + attn.rows
    print(f"  ok  50 attention replays at kv={attn.kv} identical; forward cache "
          f"stays at {fwd.kv}; a live cache would have grown to {grown}")


def test_call_counts_match_round_detail():
    """compose() never sees the calls, only round_detail. The calls it
    reconstructs from it must be exactly the calls the models received."""
    draft, target = _pair()
    for spec in (1, 3, 5, "dyga-4"):
        with _store(draft, target, log_calls=True) as store:
            out = speculative_generate(draft, target, _prompt(), k=spec,
                                       max_new_tokens=30)
        want = clear.calls_from_rounds([asdict(r) for r in out.rounds])
        assert store.calls == want, (
            f"k={spec}: reconstructed calls differ from real ones\n"
            f"  real  {store.calls[:12]}\n  recon {want[:12]}"
        )
        print(f"  ok  k={spec}: {len(want)} calls reconstructed exactly")

    with _store(draft, target, log_calls=True) as store:
        out = autoregressive_generate(target, _prompt(), max_new_tokens=12)
    want = clear.calls_from_rounds([asdict(r) for r in out.rounds], speculative=False)
    assert store.calls == want, "autoregressive calls differ"
    print(f"  ok  autoregressive: {len(want)} calls reconstructed exactly")


def test_hooks_do_not_change_output():
    draft, target = _pair()
    ref = speculative_generate(draft, target, _prompt(), k=4, max_new_tokens=24)
    with _store(draft, target):
        got = speculative_generate(draft, target, _prompt(), k=4, max_new_tokens=24)
    assert got.sequences.tolist() == ref.sequences.tolist()
    print("  ok  output identical with hooks attached")


class _TickingCounter:
    """An energy counter like NVML's: exact, but only updated every period.

    Constant ``watts`` throughout, so the true cost of a call is known:
    watts x its duration. Reading between ticks returns the last tick's
    value, which is the quantisation that wrecked the first T4 run.
    """

    supported = True

    def __init__(self, period_ms: float, watts: float, read_ms: float = 0.0):
        self.period_ms, self.watts, self.read_ms = period_ms, watts, read_ms
        self.t0 = time.perf_counter()

    def read_mj(self) -> float:
        if self.read_ms:
            time.sleep(self.read_ms / 1000)  # a slow NVML call; releases the GIL
        ms = (time.perf_counter() - self.t0) * 1000
        return (ms // self.period_ms) * self.period_ms * self.watts  # W x ms = mJ


def _busy(ms: float):
    def fn():
        end = time.perf_counter() + ms / 1000
        while time.perf_counter() < end:
            pass
    return fn


def test_tick_aligned_energy_beats_quantisation():
    """Trials cut at counter ticks must recover watts x time per call, with
    a spread far below the +/-20% a 5-update window shows when read at
    arbitrary times."""
    counter = _TickingCounter(period_ms=20, watts=50.0)
    cfg = clear.AmplifyConfig(trials=5, window_factor=10, settle_ms=50, poll_ms=1,
                              chunk_ms=2, warmup=1)
    m = clear.amplify([_busy(0.5)], "cpu", counter, period_ms=20, idle_w=30.0, cfg=cfg)
    want = 50.0 * m["time_ms"]
    err = abs(m["energy_mj"] - want) / want * 100
    assert err < 3, f"energy/call {m['energy_mj']:.4f} mJ vs true {want:.4f} ({err:.1f}% off)"
    assert m["rel_std_pct"] < 3, f"rel std {m['rel_std_pct']:.1f}% -- still quantised"
    assert abs(m["power_w"] - 50.0) < 1.5, f"power {m['power_w']:.1f} W, want 50"
    assert abs(m["dynamic_mj"] - 20.0 * m["time_ms"]) / (20.0 * m["time_ms"]) < 0.05
    assert m["ok"], m["why"]
    print(f"  ok  {m['energy_mj']:.4f} mJ/call vs true {want:.4f} ({err:.2f}% off), "
          f"rsd {m['rel_std_pct']:.2f}%, {m['trials']} trials of ~{m['window_ms']:.0f} ms")


def test_slow_counter_reads_do_not_inflate_cost():
    """On a Kaggle T4 one NVML read takes ~2 ms. Reading in the measured loop
    added that to every chunk -- +60% on a 0.1 ms component. With reads in a
    background thread, a 0.1 ms call must still measure as ~0.1 ms."""
    counter = _TickingCounter(period_ms=20, watts=50.0, read_ms=2.0)
    cfg = clear.AmplifyConfig(trials=5, window_factor=10, settle_ms=50, warmup=1,
                              poll_ms=1)
    m = clear.amplify([_busy(0.1)], "cpu", counter, period_ms=20, idle_w=30.0, cfg=cfg)
    assert m["nvml_read_ms"] >= 1.5, f"fake read took {m['nvml_read_ms']:.2f} ms"
    assert abs(m["meter_overhead_pct"]) < 5, (
        f"metered {m['time_ms']:.4f} ms vs unmetered {m['time_ms_unmetered']:.4f} ms "
        f"({m['meter_overhead_pct']:+.1f}%) -- counter reads are stalling the loop"
    )
    want = 50.0 * m["time_ms"]
    assert abs(m["energy_mj"] - want) / want < 0.03, (
        f"energy/call {m['energy_mj']:.4f} vs true {want:.4f}"
    )
    print(f"  ok  {m['nvml_read_ms']:.1f} ms reads, metering overhead "
          f"{m['meter_overhead_pct']:+.2f}%, energy within "
          f"{abs(m['energy_mj'] - want) / want * 100:.2f}%")


def test_default_polling_on_a_t4_like_counter():
    """The default 20 ms poll against a 100 ms counter, as on the T4: update
    edges are only known to within a poll interval, so the midpoint estimate
    has to keep a 1 s trial within ~2%. Then meter_check must run end to end."""
    counter = _TickingCounter(period_ms=100, watts=50.0)
    cfg = clear.AmplifyConfig(trials=3, window_factor=10, settle_ms=50, warmup=1)
    assert cfg.poll_ms == 20.0
    m = clear.amplify([_busy(0.2)], "cpu", counter, period_ms=100, idle_w=30.0, cfg=cfg)
    want = 50.0 * m["time_ms"]
    err = abs(m["energy_mj"] - want) / want * 100
    assert err < 2.5, f"energy/call {m['energy_mj']:.4f} vs true {want:.4f} ({err:.1f}% off)"

    quick = clear.AmplifyConfig(trials=2, window_factor=5, settle_ms=50, warmup=1)
    rows = clear.meter_check([_busy(0.2)], "cpu", counter, 100, 30.0,
                             polls_ms=(5.0, 50.0), cfg=quick)
    assert [r["poll_ms"] for r in rows] == [5.0, 50.0]
    for r in rows:
        assert abs(r["power_w"] - 50.0) < 2.5, f"poll {r['poll_ms']}: {r['power_w']:.1f} W"
        assert r["idle_w"] is not None and abs(r["idle_w"] - 50.0) < 5, r
    print(f"  ok  20 ms poll on a 100 ms counter: {err:.2f}% off; meter_check "
          + ", ".join(f"{r['poll_ms']:.0f} ms -> {r['power_w']:.1f} W" for r in rows))


def test_measure_and_compose_end_to_end():
    """Capture -> amplify (timing only on CPU) -> compose against a sweep.

    On CPU there is no energy counter, so the metric is time per call. The
    point is that the plumbing joins up: every call compose() needs has a
    measurement, and the breakdown sums to the total."""
    draft, target = _pair()
    prompt = _prompt()
    sweep = []
    with _store(draft, target) as store:
        for k in (1, 3):
            out = speculative_generate(draft, target, prompt, k=k, max_new_tokens=24)
            sweep.append({"strategy": f"coga-{k}", "new_tokens": out.new_tokens,
                          "round_detail": [asdict(r) for r in out.rounds]})
        out = autoregressive_generate(target, prompt, max_new_tokens=24)
        sweep.append({"strategy": "autoregressive", "new_tokens": out.new_tokens,
                      "round_detail": [asdict(r) for r in out.rounds]})

    cfg = clear.AmplifyConfig(trials=2, warmup=1, min_window_ms=2)
    measurements = []
    for key, caps in store.items():
        fns = [clear.replay_fn(store, c, "cpu") for c in caps]
        m = clear.amplify(fns, "cpu", cfg=cfg)
        assert m["time_ms"] > 0, f"{key.label}: time per call {m['time_ms']}"
        assert m["calls_per_trial"] >= len(fns), (
            f"{key.label}: {m['calls_per_trial']} calls per trial cannot cover "
            f"{len(fns)} captured inputs"
        )
        kvs = [c.kv for c in caps]
        measurements.append({
            "role": key.role, "component": key.component, "layer": key.layer,
            "rows": key.rows, "kv_bin": key.kv_bin,
            "kv_mean": sum(kvs) / len(kvs),
            "rows_mean": sum(c.rows for c in caps) / len(caps), **m,
        })
    costs = clear.Costs({"meta": {"n_layers": store.n_layers},
                         "measurements": measurements})
    assert costs.metric == "time_ms", f"no energy on CPU, yet metric is {costs.metric}"

    curve = clear.compose(costs, sweep)
    order = [r["strategy"] for r in curve]
    assert order == ["autoregressive", "coga-1", "coga-3"], f"strategies {order}"
    for r in curve:
        total = sum(r["breakdown"].values())
        assert abs(total - r["clear_per_token"]) < 1e-9 * max(1.0, total), (
            f"{r['strategy']}: breakdown {total} != total {r['clear_per_token']}"
        )
    assert not costs.approximated, f"compose had to guess: {costs.approximated}"
    checks = clear.additivity(costs)
    assert checks, "no additivity checks produced"
    bad = [c for c in checks if not (c["ratio"] and c["ratio"] > 0)]
    assert not bad, f"non-positive additivity ratios: {bad[:3]}"
    print("  ok  " + ", ".join(f"{r['strategy']} {r['clear_per_token']:.3f} ms/tok"
                               for r in curve)
          + f"; {len(checks)} additivity checks")


def _m(role, comp, rows, ms, layer=None, kv=None, rel=2.0, in_pass=None):
    """One hand-made measurement, energy at a flat 50 W.

    ``in_pass`` defaults to the amplified time, i.e. a replay that is doing
    exactly the work the pass did; pass it explicitly to model one that is not.
    """
    key = clear.Key(role, comp, layer, rows, 0 if kv is not None else None)
    return {"label": key.label, "role": role, "component": comp, "layer": layer,
            "rows": rows, "kv_bin": key.kv_bin, "kv_mean": kv,
            "rows_mean": rows if rows is not None else 10,
            "time_ms": ms, "energy_mj": ms * 50.0, "dynamic_mj": None,
            "power_w": 50.0, "rel_std_pct": rel, "trials": 5, "ok": True,
            "why": "", "meter_overhead_pct": 1.0,
            "in_pass_ms": ms if in_pass is None else in_pass, "in_pass_n": 8}


def _blob(role="target", mlp_q1_ms=0.4, quant="float16"):
    """A two-layer model whose parts add up exactly, unless mlp q1 is changed.

    The mlp's in-pass cost stays at the honest 0.4 ms, so raising
    ``mlp_q1_ms`` reproduces the fp16 failure as it really was: an
    amplified reading that no longer matches the work the pass did.
    """
    ms = [_m(role, "prefill", None, 20.0)]
    for rows in (1, 2):
        ms += [
            _m(role, "forward", rows, 2 * 1.3 + 3.21, kv=100),
            _m(role, "layer", rows, 1.3, layer=0, kv=100),
            _m(role, "attn", rows, 0.7, layer=0, kv=100),
            _m(role, "norm_in", rows, 0.1, layer=0),
            _m(role, "norm_post", rows, 0.1, layer=0),
            _m(role, "mlp", rows, mlp_q1_ms if rows == 1 else 0.4, layer=0,
               in_pass=0.4),
            _m(role, "embed", rows, 0.01), _m(role, "rope", rows, 0.1),
            _m(role, "norm", rows, 0.1), _m(role, "lm_head", rows, 3.0),
        ]
    return {"kind": "clear",
            "meta": {"n_layers": {role: 2}, "device": "test",
                     "model_pair": {"key": "pair"}, "in_pass_events": True},
            "settings": {"quant": quant, "draft_precision": "float16"},
            "replay_check": {"total": len(ms), "mismatched": 0},
            "measurements": ms}


def test_gate_catches_a_part_slower_than_its_whole():
    """The fp16 failure: mlp at rows=1 took longer alone than its whole layer.
    The gate must pass a sound file, fail that one, and fail a missing role."""
    good = clear.gate(_blob())
    assert not good["problems"], f"sound file failed the gate: {good['problems']}"
    assert not good["warnings"], f"sound file drew warnings: {good['warnings']}"

    bad = clear.gate(_blob(mlp_q1_ms=2.3))
    assert any("longer than the whole" in p for p in bad["problems"]), (
        f"a 2.3 ms mlp inside a 1.3 ms layer got through: {bad['problems']}")
    assert any("forward additivity" in p for p in bad["problems"]), (
        f"additivity of ~1.65 got through: {bad['problems']}")
    assert any("fewer rows cost more" in w for w in bad["warnings"]), (
        f"q1 slower than q2 drew no warning: {bad['warnings']}")
    assert any("inside the real pass" in p for p in bad["problems"]), (
        f"a 2.3 ms replay of a 0.4 ms component passed the anchor: {bad['problems']}")

    half = clear.gate(_blob(), roles=["draft", "target"])
    assert any("draft" in p for p in half["problems"]), (
        f"a target-only file passed as draft+target: {half['problems']}")

    blind = _blob()
    for m in blind["measurements"]:
        m.pop("in_pass_ms")
    assert any("kernel-choice check did not run" in w
               for w in clear.gate(blind)["warnings"]), (
        "a file with no in-pass timings passed without a word about it")
    print("  ok  sound file passes; slow part, bad additivity, missing role, "
          "unanchored replay and an un-anchored file all flagged")


def test_in_pass_timer_keys_match_the_store():
    """The anchor is only meaningful if it compares like with like. Running
    both over the same decode must produce the same set of keys, or a
    measurement would be checked against some other shape's cost."""
    draft, target = _pair()
    timer = clear.InPassTimer(kv_bin=8, cuda=False)
    timer.attach(draft, "draft", clear.representative_layers(3))
    timer.attach(target, "target", clear.representative_layers(3))
    with _store(draft, target) as store:
        with timer:
            speculative_generate(draft, target, _prompt(), k=3, max_new_tokens=32)
            autoregressive_generate(target, _prompt(), max_new_tokens=16)

    timed = timer.results()
    captured = {key.label for key, caps in store.items() if caps}
    assert timed, "the timer recorded nothing"
    missing = captured - set(timed)
    extra = set(timed) - captured
    assert not missing, f"{len(missing)} measured keys have no in-pass time: {sorted(missing)[:5]}"
    assert not extra, f"the timer invented keys the store never saw: {sorted(extra)[:5]}"
    assert all(v["in_pass_ms"] > 0 for v in timed.values()), "a zero in-pass time"
    assert all(v["in_pass_n"] <= timer.max_spans for v in timed.values()), "span cap leaked"
    print(f"  ok  {len(timed)} keys timed in place, exactly the keys captured")


def test_pilot_samples_enough_calls_to_be_representative():
    """A slowish call crosses the pilot's 100 ms in very few iterations, so
    the cost sizing every later stage could rest on four samples. This does
    not claim to fix any particular measurement -- warm-up cannot be the
    story, because amplify() already runs warmup*len(fns) calls before the
    pilot starts -- it just refuses to characterise a component from a
    handful of calls.
    """
    per_call, calls = 0.025, []                 # 25 ms: crosses 100 ms at n=4

    def fn():
        calls.append(1)
        time.sleep(per_call)

    def pilot_calls(min_pilot_calls):
        calls.clear()
        cfg = clear.AmplifyConfig(trials=1, min_window_ms=50, settle_ms=0,
                                  min_pilot_calls=min_pilot_calls)
        clear.amplify([fn] * 4, device="cpu", cfg=cfg)
        return len(calls)

    guarded, blind = pilot_calls(64), pilot_calls(0)
    assert guarded >= blind + 64, (
        f"the guard added only {guarded - blind} calls: {blind} without it, "
        f"{guarded} with it")
    print(f"  ok  pilot samples {guarded} calls where the duration bound alone "
          f"stopped at {blind}")


def test_chunk_never_syncs_after_every_call():
    """A chunk of 1 makes the loop sync-bound and leaves it contending with
    the counter poller, which is how a 2.6 ms call measured 9.8 ms."""
    cfg = clear.AmplifyConfig()
    assert cfg.min_chunk >= 2, "min_chunk must stop chunk collapsing to 1"
    for per_call_ms in (0.1, 2.6, 9.8, 88.0):
        chunk = max(cfg.min_chunk, round(cfg.chunk_ms / max(per_call_ms, 1e-6)))
        assert chunk >= cfg.min_chunk, f"{per_call_ms} ms -> chunk {chunk}"
    print(f"  ok  chunk floors at {cfg.min_chunk} however slow the call")


def _big_blob(lm_head_ms=2.5, lm_head_in_pass=2.5):
    """A 36-layer target sized like the real NF4 run, so an inflated lm_head
    is small against the whole forward pass instead of dwarfing it."""
    role, n = "target", 36
    ms = [_m(role, "prefill", None, 40.0)]
    for rows in (1, 2):
        ms += [
            _m(role, "forward", rows, 75.14, kv=100),
            _m(role, "layer", rows, 2.04, layer=0, kv=100),
            _m(role, "attn", rows, 1.19, layer=0, kv=100),
            _m(role, "norm_in", rows, 0.0985, layer=0),
            _m(role, "norm_post", rows, 0.0979, layer=0),
            _m(role, "mlp", rows, 0.437, layer=0),
            _m(role, "embed", rows, 0.0156), _m(role, "rope", rows, 0.127),
            _m(role, "norm", rows, 0.1005),
            _m(role, "lm_head", rows, lm_head_ms, in_pass=lm_head_in_pass),
        ]
    return {"kind": "clear",
            "meta": {"n_layers": {role: n}, "device": "test",
                     "model_pair": {"key": "pair"}, "in_pass_events": True},
            "settings": {"quant": "nf4", "draft_precision": "float16"},
            "replay_check": {"total": len(ms), "mismatched": 0},
            "measurements": ms}


def test_anchor_catches_what_the_other_checks_structurally_cannot():
    """The NF4 hole. target/lm_head was measured at 10.89 ms against 2.5 ms
    in the pass -- 4x over -- and the file passed its gate, because lm_head
    sits in no layer (so part-vs-whole has no whole to compare it with) and
    the excess is small next to a 75 ms forward (so additivity stays ~1).
    Only the in-pass anchor can see it."""
    good = clear.gate(_big_blob())
    assert not good["problems"], f"sound 36-layer file failed: {good['problems']}"

    bad = clear.gate(_big_blob(lm_head_ms=10.89))
    assert not any("longer than the whole" in p for p in bad["problems"]), (
        "part-vs-whole should have nothing to say about lm_head here")
    assert not any("additivity" in p for p in bad["problems"]), (
        f"additivity should stay inside its band: {bad['problems']}")
    assert any("inside the real pass" in p for p in bad["problems"]), (
        f"a 4x inflated lm_head passed every check: {bad['problems']}")

    # A legitimate replay runs somewhat hot -- it pays launch and sync costs
    # that overlap with other work in the real pass. A sound T4 lm_head read
    # 1.33x, so the band must not call that a failure.
    honest = clear.gate(_big_blob(lm_head_ms=2.5 * 1.33))
    assert not honest["problems"], (
        f"1.33x, which a sound measurement really showed, was failed: "
        f"{honest['problems']}")

    # A launch-bound component's in-pass span is mostly the GPU waiting for
    # the CPU, so the replay looks faster than the pass. That is an artefact,
    # and nothing that small moves a composed total: warn, do not fail.
    tiny = _big_blob()
    for m in tiny["measurements"]:
        if m["component"] == "rope":
            m["in_pass_ms"] = m["time_ms"] * 4        # 0.127 ms -> 0.25x
    out = clear.gate(tiny)
    assert not out["problems"], (
        f"a 0.127 ms rope was used to fail a run: {out['problems']}")
    assert any("launch gaps" in w for w in out["warnings"]), (
        f"an off-band small component was not even mentioned: {out['warnings']}")

    # Without CUDA events an in-pass span charges Python dispatch to the
    # component, so the same ratio must not fail a run, only flag it.
    coarse = _big_blob(lm_head_ms=10.89)
    coarse["meta"]["in_pass_events"] = False
    out = clear.gate(coarse)
    assert not any("inside the real pass" in p for p in out["problems"]), (
        "a CPU-grade in-pass time was used to fail a run")
    assert any("too coarse" in w for w in out["warnings"]), (
        f"coarse timings drew no warning at all: {out['warnings']}")
    print("  ok  4x lm_head: invisible to part-vs-whole and additivity, caught "
          "by the anchor; 1.33x allowed; CPU-grade timings only warn")


def test_gate_fails_a_component_that_is_not_reproducible_across_processes():
    """--repeat records what each component cost in fresh processes. A pick
    that differs between them means one pass's number is not a property of
    the component, so the spread has to be a problem, not a footnote."""
    blob = _blob()
    blob["repeat_spread"] = {"passes": 3, "by_label": {
        "target/mlp/q1": 4.0, "target/attn/q1/kv0": 2.5}}
    ok = clear.gate(blob)
    assert not ok["problems"], f"4% spread should pass: {ok['problems']}"
    assert any("between-process spread" in line for line in ok["lines"]), (
        f"the spread was not reported: {ok['lines']}")

    blob["repeat_spread"]["by_label"]["target/mlp/q1"] = 315.0
    drift = clear.gate(blob)
    assert any("different processes" in p for p in drift["problems"]), (
        f"a 315% between-process spread passed: {drift['problems']}")

    # A call of a few microseconds swings for reasons unrelated to kernel
    # choice, and no composed total leans on one, so the floor excludes it.
    blob["repeat_spread"]["min_ms"] = {"target/mlp/q1": 0.004}
    tiny = clear.gate(blob)
    assert not any("different processes" in p for p in tiny["problems"]), (
        f"a 4 us component was failed for moving 315%: {tiny['problems']}")
    print("  ok  small spread reported; 315% fails; 315% on a 4 us call does not")


def test_composed_total_carries_measurement_error():
    """clear_stderr is the forward measurements' spread, weighted by how often
    each is used; two totals built the same way differ by exactly nothing."""
    costs = clear.Costs(_blob())
    sweep = [{"strategy": "autoregressive", "new_tokens": 3,
              "round_detail": [{"prefix_len": 10}, {"prefix_len": 11},
                               {"prefix_len": 12}]}]
    (rec,) = clear.compose(costs, sweep)
    scale = 1.2533 * 0.02 / 5 ** 0.5
    fwd, prefill_tok = (2 * 1.3 + 3.21) * 50.0, 20.0 * 50.0 / 10
    want_total = (10 * prefill_tok + 2 * fwd) / 3
    want_se = ((10 / 3 * prefill_tok * scale) ** 2 + (2 / 3 * fwd * scale) ** 2) ** 0.5
    assert abs(rec["clear_per_token"] - want_total) < 1e-9 * want_total, (
        f"total {rec['clear_per_token']} != {want_total}")
    assert abs(rec["clear_stderr"] - want_se) < 1e-9 * want_se, (
        f"stderr {rec['clear_stderr']} != {want_se}")
    assert clear.diff_stderr(costs, rec, rec) == 0.0, "a total differs from itself"
    print(f"  ok  {rec['clear_per_token']:.2f} +/- {rec['clear_stderr']:.2f} mJ/token")


def test_merge_never_mixes_precisions_of_one_model():
    """analyze.load_clear: a target-only fp16 file replaces an NF4 target
    wholesale -- a key it lacks must not be filled in from NF4 -- while the
    draft, fp16 in both, is kept."""
    import json
    import tempfile

    import analyze

    nf4 = _blob("target", quant="nf4")
    nf4["measurements"] += _blob("draft")["measurements"]
    nf4["meta"]["n_layers"]["draft"] = 2
    fp16 = _blob("target", quant="float16")
    dropped = fp16["measurements"].pop()["label"]          # a key fp16 lacks
    for m in fp16["measurements"]:
        m["time_ms"] *= 7

    with tempfile.TemporaryDirectory() as d:
        paths = []
        for name, blob in (("nf4.json", nf4), ("fp16.json", fp16)):
            paths.append(str(Path(d) / name))
            Path(paths[-1]).write_text(json.dumps(blob))
        merged = analyze.load_clear(paths)
        same = analyze.load_clear([paths[0], paths[0]])

    target = [m for m in merged["measurements"] if m["role"] == "target"]
    assert len(target) == len(fp16["measurements"]), (
        f"{len(target)} target measurements, the fp16 file has {len(fp16['measurements'])}")
    assert dropped not in {m["label"] for m in target}, (
        f"{dropped} is missing from the fp16 file but was filled in from NF4")
    assert all(m in fp16["measurements"] for m in target), "an NF4 target value survived"
    assert sum(m["role"] == "draft" for m in merged["measurements"]) == sum(
        m["role"] == "draft" for m in nf4["measurements"]), "draft measurements lost"
    assert merged["meta"]["precision"] == {"target": "float16", "draft": "float16"}, (
        merged["meta"]["precision"])
    assert len(same["measurements"]) == len(nf4["measurements"]), (
        "merging a file with itself changed the measurement count")
    print("  ok  fp16 target replaces NF4 target wholesale; fp16 draft kept")


if __name__ == "__main__":
    tests = [
        ("capture shapes", test_capture_sees_engine_shapes),
        ("replay is exact", test_replay_reproduces_output),
        ("replay cache frozen", test_replay_does_not_grow_cache),
        ("call reconstruction", test_call_counts_match_round_detail),
        ("hooks are passive", test_hooks_do_not_change_output),
        ("tick-aligned energy", test_tick_aligned_energy_beats_quantisation),
        ("slow counter reads", test_slow_counter_reads_do_not_inflate_cost),
        ("default polling + meter check", test_default_polling_on_a_t4_like_counter),
        ("measure + compose", test_measure_and_compose_end_to_end),
        ("gate", test_gate_catches_a_part_slower_than_its_whole),
        ("pilot sample size", test_pilot_samples_enough_calls_to_be_representative),
        ("chunk floor", test_chunk_never_syncs_after_every_call),
        ("anchor vs in-pass", test_anchor_catches_what_the_other_checks_structurally_cannot),
        ("in-pass timer keys", test_in_pass_timer_keys_match_the_store),
        ("between-process spread",
         test_gate_fails_a_component_that_is_not_reproducible_across_processes),
        ("error bars", test_composed_total_carries_measurement_error),
        ("merge keeps precisions apart", test_merge_never_mixes_precisions_of_one_model),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {exc}")
    print("\n" + ("all tests passed" if not failed else f"{failed} test(s) failed"))
    sys.exit(1 if failed else 0)
