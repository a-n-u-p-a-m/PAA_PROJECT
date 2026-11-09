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
