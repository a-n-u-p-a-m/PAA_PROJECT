"""CLEAR: component-level energy via repetition (Jain et al. 2025), adapted
to the two models of speculative decoding.

Three stages, each a separate function so each can be tested on its own:

    capture   hooks on the real models during real decoding save the exact
              inputs each component saw -- hidden states, rotary tables, the
              mask, and for attention the layer's KV cache at that moment
    amplify   replay one component on its saved inputs in a continuous
              loop, first settling the GPU's power state, then cutting
              trials exactly at the energy counter's update ticks
    divide    energy / calls between two ticks is the cost of one call

``compose`` then multiplies those per-call costs by how often decoding calls
each component, which it reconstructs from a sweep's ``round_detail``. That
is why CLEAR does not have to be re-run per K: what changes with K is the
call count, not the cost of a call.

Both models are profiled, because they are not the same workload. The draft
feeds one query row per call (two after a fully accepted round, when the
bonus token and the last draft are fed together) and is called K times a
round; the target feeds K+1 rows and is called once. Attention also depends
on how much context is in the cache, so attention-bearing components are
keyed by a context-length bin as well as by row count.

Replay has one trap that would silently ruin every number: attention appends
to the KV cache. Replaying it 10,000 times against a live cache makes call
10,000 see 10,000 extra positions. Replay therefore uses ``FrozenLayerCache``,
which does exactly the concatenation a real cache does and then forgets it.
"""

from __future__ import annotations

import bisect
import math
import math
import random
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import torch

from .kv import cache_length, crop_cache, new_cache

#: components whose cost depends on context length, and so get a kv bin
#: amplified / in-pass time a measurement must land in, and the amplified
#: cost below which it is reported rather than judged. One source of truth,
#: so ``gate()`` and ``diagnose_replay.py --audit`` cannot disagree about
#: which cells are a problem. See ``InPassTimer`` for why the band is not
#: centred on 1 and why there is a floor at all.
ANCHOR_BAND = (0.7, 1.4)
ANCHOR_FLOOR_MS = 0.2

KV_DEPENDENT = ("attn", "layer", "forward")

#: per-layer components; the rest occur once per forward pass
PER_LAYER = ("layer", "norm_in", "attn", "norm_post", "mlp")

#: what a decoder layer is made of, for the additivity check
LAYER_PARTS = ("norm_in", "attn", "norm_post", "mlp")

#: what a forward pass is made of, outside its layers
MODEL_PARTS = ("embed", "rope", "norm", "lm_head")


# --------------------------------------------------------------------------
# component map
# --------------------------------------------------------------------------


def components(model: torch.nn.Module, layers: Iterable[int]) -> dict:
    """``(component, layer) -> module`` for a Llama-family causal LM.

    Qwen2, Qwen3 and Llama share this layout. Qwen3's q/k norms sit inside
    ``self_attn``, so they are counted as attention, which is where they run.
    """
    base = model.model
    out: dict[tuple[str, Optional[int]], torch.nn.Module] = {
        ("forward", None): model,
        ("embed", None): base.embed_tokens,
        ("rope", None): base.rotary_emb,
        ("norm", None): base.norm,
        ("lm_head", None): model.lm_head,
    }
    for i in layers:
        layer = base.layers[i]
        out[("layer", i)] = layer
        out[("norm_in", i)] = layer.input_layernorm
        out[("attn", i)] = layer.self_attn
        out[("norm_post", i)] = layer.post_attention_layernorm
        out[("mlp", i)] = layer.mlp
    return out


def representative_layers(n_layers: int, spec: str = "rep") -> list[int]:
    """``rep`` = first, middle, last; ``all``; ``mid``; or ``"0,5,11"``.

    Every decoder layer has the same shapes, so a few stand in for all of
    them -- but first and last are worth including because they are where
    a layer-specific difference would show, if there were one.
    """
    if spec == "all":
        return list(range(n_layers))
    if spec == "mid":
        return [n_layers // 2]
    if spec == "rep":
        return sorted({0, n_layers // 2, n_layers - 1})
    picked = sorted({int(x) for x in spec.split(",") if x.strip()})
    bad = [i for i in picked if not 0 <= i < n_layers]
    if bad:
        raise ValueError(f"layers {bad} out of range for {n_layers} layers")
    return picked


# --------------------------------------------------------------------------
# tensor plumbing
# --------------------------------------------------------------------------


class _CacheSlot:
    """Placeholder left where a live cache object was in captured kwargs."""

    def __repr__(self) -> str:
        return "<cache>"


CACHE_SLOT = _CacheSlot()


def _is_cache(obj: Any) -> bool:
    return hasattr(obj, "update") and hasattr(obj, "get_seq_length")


def _detach(obj: Any) -> Any:
    """Deep copy to CPU. A reference would be overwritten as decoding goes on."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if _is_cache(obj):
        return CACHE_SLOT
    if isinstance(obj, tuple):
        return tuple(_detach(x) for x in obj)
    if isinstance(obj, list):
        return [_detach(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _detach(v) for k, v in obj.items()}
    return obj


def _to(obj: Any, device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_to(x, device) for x in obj)
    if isinstance(obj, list):
        return [_to(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _to(v, device) for k, v in obj.items()}
    return obj


def _output_tensors(out: Any) -> list[torch.Tensor]:
    """The tensors in a module output worth comparing on replay."""
    logits = getattr(out, "logits", None)
    if logits is not None:
        return [logits]
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [t for x in out for t in _output_tensors(x)]
    return []


def _rows(args: tuple, kwargs: dict) -> int:
    """Sequence rows a call processes: dim -2 of a hidden state, -1 of ids."""
    first = next((a for a in args if isinstance(a, torch.Tensor)), None)
    if first is None:
        for name in ("hidden_states", "input_ids", "x"):
            if isinstance(kwargs.get(name), torch.Tensor):
                first = kwargs[name]
                break
    if first is None:
        raise ValueError("cannot find the input tensor of this call")
    return int(first.shape[-2] if first.dim() >= 3 else first.shape[-1])


def _find_cache(args: tuple, kwargs: dict) -> Any:
    for v in list(args) + list(kwargs.values()):
        if _is_cache(v):
            return v
    return None


def layer_kv(cache: Any, idx: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Layer ``idx``'s (keys, values) as currently held, or None if empty."""
    layers = getattr(cache, "layers", None)
    if layers is not None:
        if idx >= len(layers):
            return None
        k, v = getattr(layers[idx], "keys", None), getattr(layers[idx], "values", None)
    elif getattr(cache, "key_cache", None) is not None:
        if idx >= len(cache.key_cache):
            return None
        k, v = cache.key_cache[idx], cache.value_cache[idx]
    else:
        return None
    if k is None or k.numel() == 0:
        return None
    return k, v


class FrozenLayerCache:
    """A one-layer cache that does the real work of an update and keeps none of it.

    ``DynamicCache.update`` concatenates the new keys and values onto the
    stored ones and returns the result. This does the same concatenation --
    so the replayed call pays the same cost -- but never stores it, so the
    context length is identical on every repetition.
    """

    def __init__(self, keys: Optional[torch.Tensor], values: Optional[torch.Tensor]):
        self.keys, self.values = keys, values

    def update(self, key_states, value_states, layer_idx=None, *args, **kwargs):
        if self.keys is None:
            return key_states, value_states
        return (torch.cat([self.keys, key_states], dim=-2),
                torch.cat([self.values, value_states], dim=-2))

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0 if self.keys is None else int(self.keys.shape[-2])


def _rebuild_cache(kv: list, device) -> Any:
    """A real cache holding these per-layer (keys, values), for whole-model replay."""
    cache = new_cache()
    for i, pair in enumerate(kv):
        if pair is None:
            continue
        k, v = pair
        cache.update(k.to(device), v.to(device), i)
    return cache


# --------------------------------------------------------------------------
# stage 1: capture
# --------------------------------------------------------------------------


@dataclass
class Capture:
    role: str                       # "draft" or "target"
    component: str
    layer: Optional[int]
    rows: int                       # query rows this call processed
    kv: int                         # context cached before the call (0 if n/a)
    lm_rows: Optional[int]          # forward only: logits rows kept
    args: tuple
    kwargs: dict
    kv_state: Any = None            # layer (k, v), or a per-layer list for forward
    output: list = field(default_factory=list)


@dataclass(frozen=True)
class Key:
    """What one measurement stands for. Captures sharing a key are replayed
    round-robin inside the same amplification loop."""

    role: str
    component: str
    layer: Optional[int]
    rows: Optional[int]             # None for prefill, whose rows vary per prompt
    kv_bin: Optional[int]

    @property
    def label(self) -> str:
        parts = [self.role, self.component]
        if self.layer is not None:
            parts.append(f"L{self.layer}")
        parts.append("prefill" if self.rows is None else f"q{self.rows}")
        if self.kv_bin is not None:
            parts.append(f"kv{self.kv_bin}")
        return "/".join(parts)


def measurement_key(role: str, comp: str, idx: Optional[int], args: tuple,
                    kwargs: dict, kv_bin: int) -> tuple[Key, int, Optional[int]]:
    """Which measurement one call belongs to, with its context and logits rows.

    ``ActivationStore`` (which captures the inputs) and ``InPassTimer``
    (which times the same calls where they really run) must agree on this
    exactly: a disagreement would check an amplified measurement against a
    different shape, which is worse than not checking it. So both call this.
    """
    cache = _find_cache(args, kwargs)
    rows = _rows(args, kwargs)
    if comp == "forward":
        kv = cache_length(cache)
        lm = kwargs.get("logits_to_keep") or kwargs.get("num_logits_to_keep")
        lm_rows = int(lm) if lm else rows
        if kv == 0:
            return Key(role, "prefill", None, None, None), kv, lm_rows
        return Key(role, comp, None, rows, kv // kv_bin), kv, lm_rows
    p = layer_kv(cache, idx) if comp in ("attn", "layer") else None
    kv = 0 if p is None else int(p[0].shape[-2])
    return (Key(role, comp, idx, rows,
                kv // kv_bin if comp in KV_DEPENDENT else None), kv, None)


class ActivationStore:
    """Hooks two models and keeps a bounded, representative set of inputs.

    Every distinct key keeps at most ``per_key`` captures, chosen by
    reservoir sampling so they are spread over the whole run rather than
    being the first few calls of the first prompt. Whole-model captures carry
    every layer's cache and are much larger, so they get ``per_key_forward``.

    Calls inside a prefill are skipped for components: prefill is measured
    whole, once per role, and charged per prompt token.
    """

    def __init__(self, kv_bin: int = 128, per_key: int = 4,
                 per_key_forward: int = 1, seed: int = 0, log_calls: bool = False):
        self.kv_bin = kv_bin
        self.per_key = per_key
        self.per_key_forward = per_key_forward
        self.rng = random.Random(seed)
        self.buckets: dict[Key, list[Capture]] = {}
        self.seen: dict[Key, int] = {}
        self.calls: list[tuple[str, int, int, int]] = []
        self.log_calls = log_calls
        self.modules: dict[tuple[str, str, Optional[int]], torch.nn.Module] = {}
        self.n_layers: dict[str, int] = {}
        self.layers: dict[str, list[int]] = {}
        self._handles: list = []
        self._pending: dict[int, list[Optional[Capture]]] = {}
        self._in_prefill: dict[str, bool] = {}

    def _admit(self, key: Key) -> Optional[int]:
        """Reservoir step: the slot to (re)fill for this call, or None to skip."""
        cap = (self.per_key_forward if key.component in ("forward", "prefill")
               else self.per_key)
        n = self.seen.get(key, 0)
        self.seen[key] = n + 1
        bucket = self.buckets.setdefault(key, [])
        if len(bucket) < cap:
            bucket.append(None)  # type: ignore[arg-type]
            return len(bucket) - 1
        j = self.rng.randrange(n + 1)
        return j if j < cap else None

    # ---- hooks

    def attach(self, model: torch.nn.Module, role: str, layers: Iterable[int]) -> None:
        layers = list(layers)
        self.n_layers[role] = model.config.num_hidden_layers
        self.layers[role] = layers
        for (comp, idx), mod in components(model, layers).items():
            self.modules[(role, comp, idx)] = mod
            self._handles.append(mod.register_forward_pre_hook(
                self._pre(role, comp, idx), with_kwargs=True))
            self._handles.append(mod.register_forward_hook(
                self._post(role, comp), with_kwargs=True))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()

    def _pre(self, role: str, comp: str, idx: Optional[int]):
        def hook(module, args, kwargs):
            slot: Optional[Capture] = None
            pending = self._pending.setdefault(id(module), [])
            cache = _find_cache(args, kwargs)
            rows = _rows(args, kwargs)
            key, kv, lm_rows = measurement_key(role, comp, idx, args, kwargs,
                                               self.kv_bin)

            if comp == "forward":
                self._in_prefill[role] = kv == 0
                if self.log_calls:
                    self.calls.append((role, rows, kv, lm_rows))
                pos = self._admit(key)
                if pos is not None:
                    n = self.n_layers[role]
                    state = [None if (p := layer_kv(cache, i)) is None
                             else (_detach(p[0]), _detach(p[1])) for i in range(n)]
                    slot = Capture(role, key.component, None, rows, kv, lm_rows,
                                   _detach(args), _detach(kwargs), state)
                    self.buckets[key][pos] = slot
            elif not self._in_prefill.get(role, False):
                pos = self._admit(key)
                if pos is not None:
                    p = layer_kv(cache, idx) if comp in ("attn", "layer") else None
                    state = None if p is None else (_detach(p[0]), _detach(p[1]))
                    slot = Capture(role, comp, idx, rows, kv, None,
                                   _detach(args), _detach(kwargs), state)
                    self.buckets[key][pos] = slot
            pending.append(slot)
        return hook

    def _post(self, role: str, comp: str):
        def hook(module, args, kwargs, output):
            pending = self._pending.get(id(module))
            slot = pending.pop() if pending else None
            if slot is not None:
                slot.output = [_detach(t) for t in _output_tensors(output)]
            if comp == "forward":
                self._in_prefill[role] = False
        return hook

    # ---- results

    def items(self) -> list[tuple[Key, list[Capture]]]:
        return [(k, [c for c in v if c is not None]) for k, v in self.buckets.items()]


# --------------------------------------------------------------------------
# stage 1b: what each component costs where it really runs
# --------------------------------------------------------------------------


class InPassTimer:
    """Times every component in place, during a normal decode.

    Amplification replays one component in a tight loop, and the kernel
    cuBLAS picks for that loop is not always the one the real forward pass
    uses: selection depends on workspace and allocator state, and the pick
    is cached for the life of the process. A bad pick is therefore *stable*
    -- it survives the settle window, reads a steady rel std, and keeps
    power plausible, because time and energy inflate together.

    None of CLEAR's published gates sees that. The noise bar cannot (the
    measurement is precise, just not of the right work), %Capture only
    catches it for a component nested inside another measured one, and the
    part-vs-whole check cannot fire at all for a top-level part such as
    ``lm_head``, whose cost stays far below the whole forward pass.

    So this records the ground truth: each component's cost measured by
    hooks while the whole model runs, with CUDA events and no
    synchronisation inside the loop. ``gate()`` then compares every
    amplified measurement against it -- see ``anchor_ratio``.

    Keys match ``ActivationStore``'s exactly, because both come from
    ``measurement_key``, so each measurement is checked against its own
    shape and context bin rather than a pooled average.
    """

    def __init__(self, kv_bin: int = 128, max_spans: int = 24, cuda: bool = True):
        self.kv_bin = kv_bin
        self.max_spans = max_spans
        self.cuda = bool(cuda)
        self.spans: dict[str, list] = {}
        self.n_layers: dict[str, int] = {}
        self.modules: dict[tuple[str, str, Optional[int]], torch.nn.Module] = {}
        self._handles: list = []
        self._pending: dict[int, list] = {}
        self._in_prefill: dict[str, bool] = {}

    def _mark(self):
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev
        return time.perf_counter()

    def attach(self, model: torch.nn.Module, role: str, layers: Iterable[int]) -> None:
        self.n_layers[role] = model.config.num_hidden_layers
        for (comp, idx), mod in components(model, layers).items():
            self.modules[(role, comp, idx)] = mod
            self._handles.append(mod.register_forward_pre_hook(
                self._pre(role, comp, idx), with_kwargs=True))
            self._handles.append(mod.register_forward_hook(
                self._post(role, comp), with_kwargs=True))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()

    def _pre(self, role: str, comp: str, idx: Optional[int]):
        def hook(module, args, kwargs):
            pending = self._pending.setdefault(id(module), [])
            key, kv, _ = measurement_key(role, comp, idx, args, kwargs, self.kv_bin)
            if comp == "forward":
                self._in_prefill[role] = kv == 0
            elif self._in_prefill.get(role, False):
                pending.append(None)      # components are not measured in prefill
                return
            bucket = self.spans.setdefault(key.label, [])
            if len(bucket) >= self.max_spans:
                pending.append(None)
                return
            pending.append((bucket, self._mark()))
        return hook

    def _post(self, role: str, comp: str):
        def hook(module, args, kwargs, output):
            pending = self._pending.get(id(module))
            item = pending.pop() if pending else None
            if item is not None:
                bucket, start = item
                bucket.append((start, self._mark()))
            if comp == "forward":
                self._in_prefill[role] = False
        return hook

    def results(self) -> dict[str, dict]:
        """``{label: {"in_pass_ms": median, "in_pass_n": spans}}``."""
        if self.cuda:
            torch.cuda.synchronize()
        out: dict[str, dict] = {}
        for label, spans in self.spans.items():
            ms = []
            for start, end in spans:
                try:
                    ms.append(start.elapsed_time(end) if self.cuda
                              else (end - start) * 1000)
                except Exception:
                    continue      # an event that never ran has no elapsed time
            if ms:
                out[label] = {"in_pass_ms": statistics.median(ms),
                              "in_pass_n": len(ms)}
        return out


# --------------------------------------------------------------------------
# stage 2: replay
# --------------------------------------------------------------------------


def replay_fn(store: ActivationStore, cap: Capture, device) -> Callable[[], Any]:
    """A no-argument closure that re-executes one captured call.

    Inputs are moved to ``device`` once, here, so the loop measures the
    component and not host-to-device copies.
    """
    if cap.component == "prefill":
        model = store.modules[(cap.role, "forward", None)]
    else:
        model = store.modules[(cap.role, cap.component, cap.layer)]
    args = _to(cap.args, device)
    kwargs = _to(cap.kwargs, device)

    if cap.component in ("forward", "prefill"):
        state = cap.kv_state
        cache = _rebuild_cache(state, device) if cap.kv else None

        def run():
            kw = dict(kwargs)
            if cap.kv:
                kw["past_key_values"] = cache
                out = model(*args, **kw)
                crop_cache(cache, cap.kv)  # undo this call's append
            else:
                kw["past_key_values"] = new_cache()
                out = model(*args, **kw)
            return out
        run.cache = cache  # type: ignore[attr-defined]
        return run

    if cap.kv_state is not None or cap.component in ("attn", "layer"):
        k, v = cap.kv_state if cap.kv_state is not None else (None, None)
        frozen = FrozenLayerCache(_to(k, device), _to(v, device))
        args = tuple(frozen if a is CACHE_SLOT else a for a in args)
        kwargs = {n: frozen if a is CACHE_SLOT else a for n, a in kwargs.items()}

    def run():
        return model(*args, **kwargs)
    return run


@torch.no_grad()
def check_replay(store: ActivationStore, device) -> list[dict]:
    """Replay every capture once and compare against what it originally produced.

    If a replay does not reproduce its output, it is not the same work and
    its energy describes something else. Whole-model replays are also run
    twice to prove the cache was restored in between.
    """
    report = []
    for key, caps in store.items():
        for cap in caps:
            fn = replay_fn(store, cap, device)
            outs = [_output_tensors(fn())]
            if cap.component == "forward":
                outs.append(_output_tensors(fn()))
            worst = 0.0
            exact = True
            for got in outs:
                if len(got) != len(cap.output):
                    exact = False
                    worst = float("inf")
                    continue
                for a, b in zip(got, cap.output):
                    a = a.detach().to("cpu")
                    if a.shape != b.shape:
                        exact, worst = False, float("inf")
                    elif not torch.equal(a, b):
                        exact = False
                        worst = max(worst, float((a.float() - b.float()).abs().max()))
            cache_ok = True
            if cap.component == "forward" and getattr(fn, "cache", None) is not None:
                cache_ok = cache_length(fn.cache) == cap.kv
            report.append({"key": key.label, "kv": cap.kv, "exact": exact,
                           "max_abs_diff": worst, "cache_restored": cache_ok})
    return report


# --------------------------------------------------------------------------
# stage 3: amplify and divide
# --------------------------------------------------------------------------


@dataclass
class AmplifyConfig:
    trials: int = 5
    window_factor: float = 20.0     # each trial spans >= this many sensor periods
    min_window_ms: float = 300.0
    settle_ms: float = 1500.0       # untimed running first, so the power state
                                    # reflects this component, not the last one
    chunk_ms: float = 10.0          # sync and note the call count this often
    calls_per_window: int = 30      # window >= this many calls of a slow component
    poll_ms: float = 20.0           # gap between background NVML reads; see
                                    # _TickPoller for why not faster
    warmup: int = 3
    max_calls: int = 200_000
    tick_timeout_s: float = 5.0
    min_pilot_calls: int = 64   # so warm-up cannot dominate the pilot average
    min_chunk: int = 4          # never sync after every call: at chunk 1 the
                                # loop is bound by sync plus the counter
                                # poller rather than by the component
    empty_cache: bool = True        # same free workspace for every component,
                                    # so cuBLAS makes the same choice for all
    # Amplified / in-pass time. Not centred on 1: a part replayed on its own
    # pays launch and sync costs that overlap with other work in the real
    # pass, so somewhat above 1 is honest (a sound T4 lm_head read 1.33x).
    # Below 1 is honest too, since an in-pass span carries the hook's own
    # overhead. The failures this exists to catch were 4.2x and 4.4x.
    anchor_band: tuple = ANCHOR_BAND


def _sync(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class _TickPoller(threading.Thread):
    """Reads the energy counter in the background and logs each update.

    The measured loop must never wait on NVML itself: on a Kaggle T4 one read
    costs ~2 ms, which is longer than many components take. Reading after
    every chunk added a near-constant ~1.9 ms per chunk, inflating a 0.1 ms
    norm by 60% and a 35 ms forward pass by 10% -- enough to push the forward
    additivity to 1.2-1.6.

    It must not read too often either. Polling every 1 ms (~270 reads/s at
    2.7 ms a read) raised launch-bound components from ~35 W to ~51 W and
    overstated a target decode step by 35% against the sweep -- the reads
    themselves keep the GPU busier. Every ``poll_s`` is enough: an update's
    time is taken as the midpoint between the last read that did not see it
    and the first that did, so over a 1-2 s trial the edge error is ~1%.
    ``meter_check`` measures this side effect directly.
    """

    def __init__(self, probe, poll_s: float):
        super().__init__(daemon=True)
        self.probe, self.poll_s = probe, poll_s
        self.ticks: list[tuple[float, float]] = []   # (time, counter) at each update
        self.reads, self.read_s = 0, 0.0
        self._halt = threading.Event()

    def run(self) -> None:
        last, prev = None, None
        while not self._halt.is_set():
            t0 = time.perf_counter()
            e = self.probe.read_mj()
            t1 = time.perf_counter()
            self.reads += 1
            self.read_s += t1 - t0
            now = (t0 + t1) / 2
            if e is not None and e != last:
                if last is not None:  # the first value read is not an update edge
                    self.ticks.append(((prev + now) / 2, e))
                last = e
            prev = now
            time.sleep(self.poll_s)

    def stop(self) -> None:
        self._halt.set()
        self.join()


def _cut(ticks: list[tuple[float, float]], window_ms: float,
         trials: int) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Consecutive (start tick, end tick) pairs, each at least ``window_ms`` long."""
    out, start = [], ticks[0] if ticks else None
    for tick in ticks[1:]:
        if (tick[0] - start[0]) * 1000 >= window_ms:
            out.append((start, tick))
            start = tick
            if len(out) == trials:
                break
    return out


@torch.no_grad()
def amplify(fns: list[Callable[[], Any]], device, probe=None,
            period_ms: Optional[float] = None, idle_w: Optional[float] = None,
            cfg: Optional[AmplifyConfig] = None,
            overhead: Optional[Callable[[], Any]] = None) -> dict:
    """Run ``fns`` round-robin and return the cost of one call.

    With an energy counter this is one continuous run, cut into trials at the
    counter's own update ticks:

      settle   run untimed for ``settle_ms``. A T4 stays in a high-power state
               for a while after heavy work, so without this the first
               component after a heavy one reads up to 10x high.
      run      keep running in synced chunks, noting (time, calls done) after
               each, while a background thread logs every counter update.
      cut      a trial runs from one counter update to the first update at
               least ``window`` later; its calls are interpolated from the
               chunk log at those two instants.

    Cutting at updates is what removes quantisation. Reading at arbitrary
    times, a 500 ms window holds 5 or 6 updates of a 100 ms counter -- a
    +/-20% swing from whether one extra update landed inside -- which is what
    the first T4 run showed as rel std of either ~0.5% or 10-20%. The GPU never
    idles between trials, and the loop never waits on NVML.

    ``overhead``, if given, runs before each call (an L2 flush, say). It is
    measured on its own in the same way and subtracted.
    """
    cfg = cfg or AmplifyConfig()
    energy = probe is not None and getattr(probe, "supported", False)

    def body(n: int, start: int = 0) -> None:
        m = len(fns)
        for i in range(start, start + n):
            if overhead is not None:
                overhead()
            fns[i % m]()

    def pilot() -> float:
        """ms per call with nothing watching.

        Grows the batch until it is both long enough to time *and* big
        enough that the first calls do not dominate the average. Duration
        alone is not enough: a slowish call reaches 100 ms in very few
        iterations, and those early ones are the expensive ones. lm_head
        allocates a vocabulary-wide output that the caching allocator only
        returns cheaply after a few calls, so a pilot that stopped at 16
        calls read 6.8 ms for work the real pass does in 2.6 ms -- and
        every later stage inherited it, because ``chunk`` is sized from
        this, and a chunk of 1 syncs after every call and then contends
        with the counter poller.

        A call slow enough that a full second of it has already run is
        warm whatever the count, so ``dt >= 1000`` releases the count
        requirement and prefill-sized work does not pay for it.
        """
        n = len(fns)
        while True:
            t0 = time.perf_counter()
            body(n)
            _sync(device)
            dt = (time.perf_counter() - t0) * 1000
            warm = n >= cfg.min_pilot_calls or dt >= 1000
            if (dt >= 100 and warm) or n >= cfg.max_calls:
                return dt / n
            n *= 4

    # Which kernel cuBLAS picks depends on how much workspace is free, so
    # hand every component the same allocator state. Otherwise the pick
    # drifts as the run fragments memory and a component measured late is
    # not comparable with one measured early.
    if cfg.empty_cache and str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    for _ in range(cfg.warmup):
        body(len(fns))
    _sync(device)
    per_call_ms = first_pilot = pilot()
    settled_pilot = None

    if not energy:
        return _timing_only(body, per_call_ms, len(fns), device, cfg)

    chunk = max(cfg.min_chunk, round(cfg.chunk_ms / max(per_call_ms, 1e-6)))
    window = max(cfg.min_window_ms, cfg.window_factor * (period_ms or 100.0),
                 cfg.calls_per_window * chunk * per_call_ms)

    done = 0
    t_end = time.perf_counter() + cfg.settle_ms / 1000
    while time.perf_counter() < t_end:
        body(chunk, done)
        _sync(device)
        done += chunk

    # Time it again now it has been running for settle_ms. A pick that was
    # slow only because the first calls paid to set it up reads faster here;
    # one that is genuinely wrong does not move, which is the signal the
    # anchor check needs. Keep the better of the two.
    settled_pilot = pilot()
    if settled_pilot < per_call_ms:
        per_call_ms = settled_pilot
        chunk = max(cfg.min_chunk, round(cfg.chunk_ms / max(per_call_ms, 1e-6)))

    poller = _TickPoller(probe, cfg.poll_ms / 1000)
    poller.start()
    marks = [(time.perf_counter(), done)]
    give_up = time.perf_counter() + cfg.tick_timeout_s + 2 * cfg.trials * window / 1000
    try:
        while True:
            body(chunk, done)
            _sync(device)
            done += chunk
            marks.append((time.perf_counter(), done))
            if len(_cut(poller.ticks, window, cfg.trials)) >= cfg.trials:
                break
            if marks[-1][0] > give_up:
                raise RuntimeError("energy counter stopped updating")
        body(chunk, done)  # one more mark after the last tick, so it is bracketed
        _sync(device)
        done += chunk
        marks.append((time.perf_counter(), done))
    finally:
        poller.stop()

    times = [t for t, _ in marks]

    def calls_at(t: float) -> float:
        i = min(max(bisect.bisect_left(times, t), 1), len(marks) - 1)
        (t0, c0), (t1, c1) = marks[i - 1], marks[i]
        return c0 + (c1 - c0) * (t - t0) / (t1 - t0) if t1 > t0 else float(c1)

    rows = [(e1 - e0, (t1 - t0) * 1000, calls_at(t1) - calls_at(t0))
            for (t0, e0), (t1, e1) in _cut(poller.ticks, window, cfg.trials)]

    res = _summarise(rows, idle_w, period_ms)
    res["time_ms_unmetered"] = per_call_ms
    # Both pilots, so a reading that never improved can be told apart from
    # one that only needed warming up.
    res["pilot_ms"] = {"first": first_pilot, "settled": settled_pilot}
    res["meter_overhead_pct"] = (res["time_ms"] / per_call_ms - 1) * 100
    res["nvml_read_ms"] = poller.read_s / max(poller.reads, 1) * 1000
    if overhead is not None:
        base = amplify([overhead], device, probe, period_ms, idle_w, cfg)
        for k in ("time_ms", "energy_mj", "dynamic_mj"):
            if res.get(k) is not None and base.get(k) is not None:
                res[k] -= base[k]
        res["overhead_subtracted"] = {k: base.get(k) for k in
                                      ("time_ms", "energy_mj", "dynamic_mj")}
    return res


@torch.no_grad()
def meter_check(fns: list[Callable[[], Any]], device, probe, period_ms: Optional[float],
                idle_w: Optional[float], polls_ms=(2.0, 20.0, 100.0),
                cfg: Optional[AmplifyConfig] = None) -> list[dict]:
    """Does reading the counter change what it reads?

    For each polling interval: idle power with the poller running (nothing
    else on the GPU), then one reference workload measured with that poller.
    If the readings move with the polling rate, the meter is perturbing the
    measurement, and the slowest rate that still resolves updates is right.
    """
    cfg = cfg or AmplifyConfig()
    out = []
    for poll in polls_ms:
        _sync(device)
        poller = _TickPoller(probe, poll / 1000)
        poller.start()
        time.sleep(3.0)
        poller.stop()
        tk = poller.ticks
        idle = ((tk[-1][1] - tk[0][1]) / ((tk[-1][0] - tk[0][0]) * 1000)
                if len(tk) >= 2 and tk[-1][0] > tk[0][0] else None)  # mJ/ms = W
        c = AmplifyConfig(**{**cfg.__dict__, "poll_ms": poll, "trials": 3})
        m = amplify(fns, device, probe, period_ms, idle_w, c)
        out.append({"poll_ms": poll, "reads_per_s": poller.reads / 3.0,
                    "idle_w": idle, "power_w": m["power_w"], "energy_mj": m["energy_mj"],
                    "time_ms": m["time_ms"], "rel_std_pct": m["rel_std_pct"],
                    "meter_overhead_pct": m["meter_overhead_pct"]})
    return out


def _timing_only(body, per_call_ms: float, n_inputs: int, device,
                 cfg: AmplifyConfig) -> dict:
    """No energy counter (CPU, P100): fixed-size timed trials, each covering
    every captured input at least once."""
    n = int(min(cfg.max_calls, max(n_inputs, math.ceil(cfg.min_window_ms / max(per_call_ms, 1e-6)))))
    times = []
    for _ in range(cfg.trials):
        _sync(device)
        t0 = time.perf_counter()
        body(n)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    t_med = statistics.median(times)
    spread = (statistics.stdev(times) / statistics.mean(times) * 100
              if len(times) > 1 else None)
    return {"calls_per_trial": n, "trials": len(times), "window_ms": round(t_med, 2),
            "time_ms": t_med / n, "energy_mj": None, "dynamic_mj": None,
            "power_w": None, "rel_std_pct": spread, "zero_trials": 0,
            "ok": True, "why": ""}


def _summarise(rows: list[tuple[float, float, int]], idle_w: Optional[float],
               period_ms: Optional[float]) -> dict:
    """``rows`` = (energy mJ, time ms, calls) per tick-aligned trial."""
    e_call = [e / n for e, _, n in rows]
    t_call = [t / n for _, t, n in rows]
    t_med = statistics.median(t for _, t, _ in rows)
    out: dict[str, Any] = {
        "calls_per_trial": round(statistics.mean(n for _, _, n in rows)),
        "trials": len(rows),
        "window_ms": round(t_med, 2),
        "time_ms": statistics.median(t_call),
        "energy_mj": statistics.median(e_call),
        "dynamic_mj": None,
        "power_w": statistics.median(e / t for e, t, _ in rows),  # mJ/ms = W
        "rel_std_pct": None,
        "zero_trials": sum(1 for e, _, _ in rows if e <= 0),
        "ok": True,
        "why": "",
    }
    if len(e_call) > 1 and statistics.mean(e_call) > 0:
        out["rel_std_pct"] = statistics.stdev(e_call) / statistics.mean(e_call) * 100
    if idle_w:
        out["dynamic_mj"] = statistics.median((e - idle_w * t) / n for e, t, n in rows)
    problems = []
    if out["zero_trials"]:
        problems.append(f"{out['zero_trials']} trial(s) read 0 mJ")
    if period_ms and t_med < 3 * period_ms:
        problems.append(f"window {t_med:.0f} ms < 3 sensor periods")
    if out["rel_std_pct"] is not None and out["rel_std_pct"] >= 10:
        problems.append(f"rel std {out['rel_std_pct']:.1f}% >= 10%")
    out["ok"] = not problems
    out["why"] = "; ".join(problems)
    return out


def l2_flusher(device) -> Optional[Callable[[], Any]]:
    """Overwrite twice the L2 so the next call starts with cold caches.

    Replaying one small component back to back keeps its weights resident in
    L2, which real decoding -- alternating draft and target -- does not. The
    flush is subtracted afterwards, so it changes cache state, not the result.
    """
    if not str(device).startswith("cuda"):
        return None
    size = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 0) or (6 << 20)
    buf = torch.empty(2 * size, dtype=torch.uint8, device=device)
    return lambda: buf.zero_()


# --------------------------------------------------------------------------
# composition: per-call cost x call counts
# --------------------------------------------------------------------------


def calls_from_rounds(rounds: list[dict], speculative: bool = True) -> list[tuple]:
    """Every forward call a generation made, as ``(role, rows, kv, lm_rows)``.

    Reconstructed from ``round_detail``, using the engine's bookkeeping:

      verify, round r>0   rows K+1, kv n-1, lm K+1
      draft,  call 0      rows 1, kv n-1 -- or rows 2, kv n-2 when the previous
                          round accepted all its drafts, because d_K was never
                          fed back and goes in alongside the bonus token
      draft,  call j>0    rows 1, kv n+j-1
      round 0             both models prefill from an empty cache

    ``tests/test_clear.py`` checks this against the calls the hooks actually see.
    """
    calls = []
    if not speculative:
        for i, r in enumerate(rounds):
            n = r["prefix_len"]
            calls.append(("target", n, 0, 1) if i == 0 else ("target", 1, n - 1, 1))
        return calls

    prev = None
    for i, r in enumerate(rounds):
        n, k = r["prefix_len"], r["proposed"]
        for j in range(k):
            if j == 0:
                if i == 0:
                    calls.append(("draft", n, 0, 1))
                else:
                    q = 2 if prev["accepted"] == prev["proposed"] else 1
                    calls.append(("draft", q, n - q, 1))
            else:
                calls.append(("draft", 1, n + j - 1, 1))
        if i == 0:
            calls.append(("target", n + k, 0, k + 1))
        else:
            calls.append(("target", k + 1, n - 1, k + 1))
        prev = r
    return calls


class Costs:
    """Lookup of measured per-call cost, interpolating over context length."""

    def __init__(self, blob: dict, metric: Optional[str] = None):
        self.meta = blob.get("meta", {})
        ms = [m for m in blob["measurements"]]
        # Total energy by default, not energy above idle. Most components
        # are launch-bound and leave the GPU near idle, so their real cost is
        # mostly idle draw x time; subtracting idle leaves only noise there,
        # and the sweep's NVML reading to validate against is total too.
        if metric is None:
            for metric in ("energy_mj", "dynamic_mj", "time_ms"):
                if ms and all(m.get(metric) is not None for m in ms):
                    break
        self.metric = metric
        self.unit = {"dynamic_mj": "mJ above idle", "energy_mj": "mJ",
                     "time_ms": "ms"}[metric]
        self.approximated: dict[str, int] = {}
        self.table: dict[tuple, list[tuple[float, float]]] = {}
        #: standard error of each table entry, in the same order as ``table``
        self.err: dict[tuple, list[float]] = {}
        raw: dict[tuple, list[tuple[float, float, float]]] = {}
        per_token: dict[str, list[tuple[float, float]]] = {}
        for m in ms:
            v = m[self.metric]
            se = self._stderr(m, v)
            if m["component"] == "prefill":
                # prefill is charged per prompt token; prompts differ in length
                per_token.setdefault(m["role"], []).append(
                    (v / m["rows_mean"], se / m["rows_mean"]))
                continue
            k = (m["role"], m["component"], m["layer"], m["rows"])
            raw.setdefault(k, []).append((m["kv_mean"] or 0.0, v, se))
        for k, pts in raw.items():
            pts.sort()
            self.table[k] = [(kv, v) for kv, v, _ in pts]
            self.err[k] = [se for _, _, se in pts]
        self.prefill = {r: statistics.mean(v for v, _ in pts)
                        for r, pts in per_token.items()}
        self.prefill_err = {r: math.sqrt(sum(se * se for _, se in pts)) / len(pts)
                            for r, pts in per_token.items()}

    def _stderr(self, m: dict, v: float) -> float:
        """Standard error of one measurement, from its trial-to-trial spread.

        ``rel_std_pct`` is the spread of energy per call across trials (of
        time per call when there is no counter). Energy above idle has the
        same absolute spread as total energy, not the same relative one. The
        reported value is a median, whose standard error is ~1.2533 sigma/sqrt(n).
        """
        rel, n = m.get("rel_std_pct"), m.get("trials") or 0
        if rel is None or n < 2:
            return 0.0
        base = v
        if self.metric == "dynamic_mj" and m.get("energy_mj") is not None:
            base = m["energy_mj"]
        return 1.2533 * rel / 100.0 * abs(base) / math.sqrt(n)

    def layers(self, role: str) -> list[int]:
        return sorted({k[2] for k in self.table if k[0] == role and k[2] is not None})

    def n_layers(self, role: str) -> int:
        return self.meta["n_layers"][role]

    def get(self, role: str, comp: str, layer: Optional[int], rows: int,
            kv: int) -> float:
        key = (role, comp, layer, rows)
        if key not in self.table:
            near = [k for k in self.table if k[:3] == key[:3]]
            if not near:
                raise KeyError(f"no measurement for {role}/{comp}"
                               + (f"/L{layer}" if layer is not None else ""))
            key = min(near, key=lambda k: abs(k[3] - rows))
            tag = f"{role}/{comp} q{rows}->q{key[3]}"
            self.approximated[tag] = self.approximated.get(tag, 0) + 1
        pts = self.table[key]
        if comp not in KV_DEPENDENT or len(pts) == 1:
            return statistics.mean(v for _, v in pts)
        if kv <= pts[0][0]:
            return pts[0][1]
        if kv >= pts[-1][0]:
            return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= kv <= x1:
                return y0 if x1 == x0 else y0 + (y1 - y0) * (kv - x0) / (x1 - x0)
        return pts[-1][1]

    def total_terms(self, role: str, rows: int, kv: int) -> list[tuple[tuple, float]]:
        """``call()["total"]`` as a weighted sum of measurements.

        Returns ``(measurement id, weight)`` pairs, mirroring the lookup and
        context interpolation in ``get``. ``compose`` adds these up so the
        measurement uncertainty can be carried into the per-K totals.
        """
        if kv == 0:
            return [(("prefill", role), float(rows))]
        key = (role, "forward", None, rows)
        if key not in self.table:
            near = [k for k in self.table if k[:3] == key[:3]]
            if not near:
                raise KeyError(f"no measurement for {role}/forward")
            key = min(near, key=lambda k: abs(k[3] - rows))
        pts = self.table[key]
        if len(pts) == 1 or kv <= pts[0][0]:
            return [((key, 0), 1.0)]
        if kv >= pts[-1][0]:
            return [((key, len(pts) - 1), 1.0)]
        for i, ((x0, _), (x1, _)) in enumerate(zip(pts, pts[1:])):
            if x0 <= kv <= x1:
                if x1 == x0:
                    return [((key, i), 1.0)]
                t = (kv - x0) / (x1 - x0)
                return [((key, i), 1.0 - t), ((key, i + 1), t)]
        return [((key, len(pts) - 1), 1.0)]

    def stderr(self, ident: tuple) -> float:
        """Standard error of one measurement id from ``total_terms``."""
        if ident[0] == "prefill":
            return self.prefill_err.get(ident[1], 0.0)
        key, i = ident
        return self.err[key][i]

    def combined_stderr(self, weights: dict) -> float:
        """Standard error of ``sum(weight x measurement)``, the measurements
        being independent of each other."""
        return math.sqrt(sum((w * self.stderr(i)) ** 2 for i, w in weights.items()))

    def layer_mean(self, role: str, comp: str, rows: int, kv: int) -> float:
        """Mean over measured layers, standing in for any one layer."""
        return statistics.mean(self.get(role, comp, l, rows, kv)
                               for l in self.layers(role))

    def call(self, role: str, rows: int, kv: int, lm_rows: int) -> dict:
        """Cost of one forward call, total and split by component.

        The total is the whole-model measurement: it is what the hardware
        actually did. The split is CLEAR's per-component numbers, and
        ``unattributed`` is the difference -- launch overhead, residual adds,
        mask construction, whatever the components do not cover.
        """
        if kv == 0:
            per_tok = self.prefill.get(role)
            if per_tok is None:
                raise KeyError(f"no prefill measurement for {role}")
            return {"total": per_tok * rows, "prefill": per_tok * rows}
        n = self.n_layers(role)
        split = {
            "embed": self.get(role, "embed", None, rows, kv),
            "rope": self.get(role, "rope", None, rows, kv),
            "norm": self.get(role, "norm", None, rows, kv),
            "lm_head": self.get(role, "lm_head", None, lm_rows, kv),
        }
        for part in LAYER_PARTS:
            split[part] = n * self.layer_mean(role, part, rows, kv)
        total = self.get(role, "forward", None, rows, kv)
        split["unattributed"] = total - sum(split.values())
        return {"total": total, **split}


def compose(costs: Costs, rows: list[dict]) -> list[dict]:
    """Per strategy: CLEAR cost per emitted token, total and broken down.

    ``rows`` are a sweep's per-prompt rows (from ``run_baseline.py``). The
    per-prompt result is averaged the same way ``analyze.py`` averages
    everything else, so the columns are comparable.
    """
    by: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("round_detail"):
            by.setdefault(r["strategy"], []).append(r)

    out = []
    for name, group in by.items():
        spec = name != "autoregressive"
        per_tok, parts = [], {}
        weights: dict[tuple, float] = {}
        for r in group:
            total = 0.0
            split: dict[str, float] = {}
            scale = 1.0 / ((r["new_tokens"] or 1) * len(group))
            for role, q, kv, lm in calls_from_rounds(r["round_detail"], spec):
                c = costs.call(role, q, kv, lm)
                total += c["total"]
                for ident, w in costs.total_terms(role, q, kv):
                    weights[ident] = weights.get(ident, 0.0) + w * scale
                for k, v in c.items():
                    if k != "total":
                        split[f"{role}.{k}"] = split.get(f"{role}.{k}", 0.0) + v
            tokens = r["new_tokens"] or 1
            per_tok.append(total / tokens)
            for k, v in split.items():
                parts.setdefault(k, []).append(v / tokens)
        out.append({
            "strategy": name,
            "k": int(name.split("-")[1]) if name.startswith("coga-") else None,
            "clear_per_token": statistics.mean(per_tok),
            # Repeatability of the forward measurements only. It leaves out
            # systematic error and prompt sampling, and totals at different K
            # share measurements, so compare two of them with diff_stderr().
            "clear_stderr": costs.combined_stderr(weights),
            "weights": weights,
            "breakdown": {k: statistics.mean(v) for k, v in parts.items()},
            "n_prompts": len(group),
        })
    out.sort(key=lambda r: (r["strategy"] != "autoregressive", r["k"] is None,
                            r["k"] or 0, r["strategy"]))
    return out


def diff_stderr(costs: Costs, a: dict, b: dict) -> float:
    """Standard error of the difference between two ``compose`` records.

    Their totals are built from overlapping measurements, so the errors are
    correlated and the per-record ``clear_stderr`` values cannot simply be
    combined; the shared part cancels here.
    """
    wa, wb = a["weights"], b["weights"]
    return costs.combined_stderr({i: wa.get(i, 0.0) - wb.get(i, 0.0)
                                  for i in set(wa) | set(wb)})


def additivity(costs: Costs) -> list[dict]:
    """Do the parts add up to the whole?

    Two checks, at each (role, rows, kv bin) measured: a decoder layer
    against its four parts, and a forward pass against all of its parts.
    A sum *above* the whole is common -- each part replayed alone pays its
    own launch and sync cost, which overlaps in the real pass -- and a sum
    far below means something is missing from the component map.
    """
    out = []
    for (role, comp, layer, rows), pts in costs.table.items():
        for kv, whole in pts:
            if comp == "layer":
                parts = sum(costs.get(role, p, layer, rows, kv) for p in LAYER_PARTS)
                out.append({"check": f"{role} L{layer} layer", "rows": rows,
                            "kv": kv, "whole": whole, "parts": parts,
                            "ratio": parts / whole if whole else None})
            elif comp == "forward":
                c = costs.call(role, rows, int(kv), rows if role == "target" else 1)
                parts = c["total"] - c["unattributed"]
                out.append({"check": f"{role} forward", "rows": rows, "kv": kv,
                            "whole": whole, "parts": parts,
                            "ratio": parts / whole if whole else None})
    out.sort(key=lambda r: (r["check"], r["rows"], r["kv"]))
    return out


def gate(blob: dict, roles: Optional[Iterable[str]] = None,
         max_power_w: Optional[float] = None,
         forward_range: tuple[float, float] = (0.8, 1.25),
         layer_range: tuple[float, float] = (0.8, 1.25),
         anchor_band: tuple[float, float] = ANCHOR_BAND,
         anchor_floor_ms: float = ANCHOR_FLOOR_MS,
         max_spread_pct: float = 25.0,
         spread_floor_ms: float = 0.05) -> dict:
    """Is this ``run_clear.py`` output fit to compose from?

    Returns ``{"lines", "problems", "warnings"}``. A problem means the
    numbers must not be used; a warning means a figure that does not enter
    the composed total looks off and needs explaining before it is quoted.

    Beyond noise and additivity, four checks look for a replay that is not
    doing the work the real pass did. Three need only time, no counter:
    a part replayed alone must not take longer than the whole that contains
    it; fewer rows should not take much longer than more rows; and every
    component's amplified cost must match what it costs measured in place
    by ``InPassTimer`` (``anchor_band``) -- the one check that catches a
    kernel pick that is wrong but stable, which the others cannot see. The
    fourth, if ``--repeat`` was used, is whether a component costs the same
    in a fresh process (``max_spread_pct``).
    """
    ms = blob["measurements"]
    problems: list[str] = []
    warnings: list[str] = []
    lines: list[str] = []

    present = sorted({m["role"] for m in ms})
    want = sorted(roles) if roles else present
    missing = [r for r in want if r not in present]
    if missing:
        problems.append(f"no measurements for {', '.join(missing)}")
    lines.append(f"roles measured:         {', '.join(present) or 'none'}")
    if not ms:
        return {"lines": lines, "problems": problems or ["no measurements"],
                "warnings": warnings}

    rc = blob.get("replay_check") or {}
    if rc.get("mismatched"):
        problems.append(f"{rc['mismatched']} replays did not reproduce")
    if rc:
        lines.append(f"replays exact:          {rc['total'] - rc['mismatched']}/{rc['total']}")

    bad = [m for m in ms if not m["ok"]]
    if len(bad) > len(ms) * 0.1:
        problems.append(f"{len(bad)}/{len(ms)} measurements outside the 10% bar")
    lines.append(f"inside 10% bar:         {len(ms) - len(bad)}/{len(ms)}")

    over = [m["meter_overhead_pct"] for m in ms if m.get("meter_overhead_pct") is not None]
    if over:
        med = statistics.median(over)
        if abs(med) > 5:
            problems.append(f"metering adds {med:+.1f}% to call time")
        lines.append(f"metering overhead:      {med:+.1f}% median")

    if max_power_w:
        hot = [m.get("label", m["component"]) for m in ms
               if (m.get("power_w") or 0) > max_power_w]
        if hot:
            problems.append(f"impossible power (>{max_power_w:.0f} W): {hot[:3]}")

    # ---- a part must not outlast its whole; rows should not run backwards
    t = Costs(blob, "time_ms")
    slow, backwards = [], []
    by_comp: dict[tuple, dict[int, float]] = {}
    for (role, comp, layer, rows), pts in t.table.items():
        by_comp.setdefault((role, comp, layer), {})[rows] = statistics.mean(
            v for _, v in pts)
    for (role, comp, layer), by_rows in by_comp.items():
        where = f"{role}/{comp}" + (f"/L{layer}" if layer is not None else "")
        if comp in LAYER_PARTS:
            whole = by_comp.get((role, "layer", layer), {})
        elif comp in MODEL_PARTS:
            whole = by_comp.get((role, "forward", None), {})
        else:
            whole = {}
        for rows, v in sorted(by_rows.items()):
            w = whole.get(rows)
            if w and v > 1.1 * w:
                slow.append(f"{where}/q{rows} {v:.3f} ms alone vs {w:.3f} ms for "
                            f"the whole {'layer' if comp in LAYER_PARTS else 'forward'}")
        if comp not in KV_DEPENDENT:
            seq = sorted(by_rows.items())
            for (r0, v0), (r1, v1) in zip(seq, seq[1:]):
                if v0 > 1.5 * v1:
                    backwards.append(f"{where} q{r0} {v0:.3f} ms > q{r1} {v1:.3f} ms")
    if slow:
        problems.append("a part replayed alone takes longer than the whole that "
                        "contains it, so the replay is not the work the pass did: "
                        + "; ".join(slow[:4])
                        + (f" (+{len(slow) - 4} more)" if len(slow) > 4 else ""))
    if backwards:
        warnings.append("fewer rows cost more than more rows: " + "; ".join(backwards[:4])
                        + (f" (+{len(backwards) - 4} more)" if len(backwards) > 4 else ""))
    lines.append(f"part slower than whole: {len(slow)}")

    # ---- the anchor: is the replay doing the work the pass did?
    # The only check that catches a stable-but-wrong kernel pick. The noise
    # bar cannot (such a measurement is precise), power cannot (time and
    # energy inflate together, so W stays plausible), and part-vs-whole
    # cannot reach a top-level part like lm_head, which stays far below the
    # whole forward pass however inflated it is. Time only, no counter.
    lo_a, hi_a = anchor_band
    anchored = [m for m in ms if m.get("in_pass_ms") and m.get("time_ms")]
    if anchored:
        ratios = [m["time_ms"] / m["in_pass_ms"] for m in anchored]
        say = lambda m: (f"{m.get('label', m['component'])} "
                         f"{m['time_ms'] / m['in_pass_ms']:.2f}x "
                         f"({m['time_ms']:.3f} ms amplified vs "
                         f"{m['in_pass_ms']:.3f} ms in the pass)")
        bad = [m for m in anchored
               if not lo_a <= m["time_ms"] / m["in_pass_ms"] <= hi_a]
        # An in-pass span is bracketed on the stream, so for a component
        # small enough to be launch-bound it also covers the gaps where the
        # GPU waits for the CPU. That reads as the replay being faster than
        # the pass, which is an artefact, not a finding. Below the floor,
        # say so rather than fail: nothing that small moves a composed total.
        #
        # The floor is on the amplified time, not the in-pass time: an
        # inflated in-pass reading is exactly what this is protecting
        # against, so letting it clear its own floor would exempt the cases
        # worth catching.
        off = [say(m) for m in bad if m["time_ms"] >= anchor_floor_ms]
        small = [say(m) for m in bad if m["time_ms"] < anchor_floor_ms]
        lines.append(f"anchor vs in-pass:      {min(ratios):.2f} .. {max(ratios):.2f}"
                     f"  ({len(anchored)}/{len(ms)} anchored, {len(off)} off)")
        if small:
            warnings.append(
                f"{len(small)} component(s) under {anchor_floor_ms} ms are outside "
                "the anchor band, where an in-pass span is mostly launch gaps: "
                + "; ".join(small[:3]))
        if off and not blob.get("meta", {}).get("in_pass_events"):
            # Without CUDA events an in-pass span is bounded by perf_counter
            # around a hook, which charges Python dispatch to the component
            # and swamps anything this small. Report it, do not judge on it.
            warnings.append(
                f"{len(off)} measurements are outside the anchor band, but this "
                "file's in-pass times were not taken with CUDA events, so they "
                "are too coarse to fail a run on: " + "; ".join(off[:2]))
        elif off:
            problems.append(
                "amplified cost does not match what the component costs inside "
                f"the real pass (outside {lo_a} .. {hi_a}), so the replay is not "
                "running the kernel the pass runs: " + "; ".join(off[:4])
                + (f" (+{len(off) - 4} more)" if len(off) > 4 else ""))
    else:
        warnings.append("no in-pass timings in this file, so the kernel-choice "
                        "check did not run: re-measure with InPassTimer attached "
                        "before quoting a component split")

    # ---- repeatability across processes, if --repeat was used
    rep = blob.get("repeat_spread") or {}
    by_label = rep.get("by_label") or {}
    if by_label:
        ranked = sorted(by_label.items(), key=lambda kv: -kv[1])
        lines.append(f"between-process spread: {ranked[-1][1]:.1f}% .. "
                     f"{ranked[0][1]:.1f}%  over {rep.get('passes', '?')} processes")
        # A call of a few microseconds swings wildly for reasons that have
        # nothing to do with kernel choice, and no composed total leans on
        # one. Judge only components big enough to matter.
        floor = rep.get("min_ms") or {}
        drifted = [f"{lab} {v:.0f}%" for lab, v in ranked
                   if v > max_spread_pct
                   and floor.get(lab, float("inf")) >= spread_floor_ms]
        if drifted:
            problems.append(
                "the same component costs different amounts in different "
                f"processes, by more than {max_spread_pct:.0f}%, so one pass's "
                "number is not reproducible: " + "; ".join(drifted[:4])
                + (f" (+{len(drifted) - 4} more)" if len(drifted) > 4 else ""))

    # ---- additivity
    try:
        checks = additivity(Costs(blob))
    except KeyError as exc:
        problems.append(f"additivity could not be computed: {exc}")
        checks = []
    fwd = [c["ratio"] for c in checks if c["check"].endswith("forward") and c["ratio"]]
    lay = [c["ratio"] for c in checks if c["check"].endswith("layer") and c["ratio"]]
    if fwd:
        lo, hi = forward_range
        if not all(lo <= r <= hi for r in fwd):
            problems.append(f"forward additivity {min(fwd):.2f} .. {max(fwd):.2f}, "
                            f"outside {lo} .. {hi}")
        lines.append(f"forward additivity:     {min(fwd):.2f} .. {max(fwd):.2f}")
    elif checks or not problems:
        problems.append("no forward additivity checks: forward or component "
                        "measurements are missing")
    if lay:
        lo, hi = layer_range
        if not all(lo <= r <= hi for r in lay):
            # The composed total is built from forward measurements, so this
            # does not block it. It does mean the per-component split inside
            # a layer is off by this much, and if the forward ratio is still
            # near 1 something outside the layers is over-counted to match.
            warnings.append(f"layer additivity {min(lay):.2f} .. {max(lay):.2f}, "
                            f"outside {lo} .. {hi}: the component split is not "
                            f"yet trustworthy at this level")
        lines.append(f"layer additivity:       {min(lay):.2f} .. {max(lay):.2f}")
    return {"lines": lines, "problems": problems, "warnings": warnings}
