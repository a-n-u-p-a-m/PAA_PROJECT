"""KV-cache compatibility helpers.

The HuggingFace cache API has been refactored several times (legacy tuples ->
``DynamicCache`` with ``key_cache``/``value_cache`` lists -> ``cache.layers[i].keys``).
Speculative decoding needs two operations the public API does not guarantee
across versions: ask a cache how long it is, and truncate it after a rollback.
Everything version-specific is isolated here.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

import torch


def new_cache() -> Any:
    """Return a fresh empty cache object, or ``None`` on very old versions."""
    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache()
    except Exception:  # pragma: no cover - ancient transformers
        return None


def cache_length(cache: Any) -> int:
    """Number of token positions currently held in ``cache``."""
    if cache is None:
        return 0

    get_len = getattr(cache, "get_seq_length", None)
    if callable(get_len):
        try:
            return int(get_len())
        except Exception:
            pass

    layers = getattr(cache, "layers", None)
    if layers:
        keys = getattr(layers[0], "keys", None)
        if keys is not None:
            return int(keys.shape[-2])

    key_cache = getattr(cache, "key_cache", None)
    if key_cache:
        return int(key_cache[0].shape[-2])

    if isinstance(cache, (tuple, list)) and len(cache) and len(cache[0]):
        return int(cache[0][0].shape[-2])

    return 0


def crop_cache(cache: Any, max_length: int) -> Any:
    """Truncate ``cache`` to its first ``max_length`` positions.

    This is the rollback primitive: when the target model rejects a draft
    token, every cache entry from that position onward was computed from a
    prefix the accepted sequence no longer contains, and must be discarded.
    """
    if cache is None or max_length < 0:
        return cache
    current = cache_length(cache)
    if current <= max_length:
        return cache

    crop = getattr(cache, "crop", None)
    if callable(crop):
        try:
            # negative = "remove this many from the end"; accepted by every
            # version, where the positive form is deprecated from v5.18
            crop(-(current - max_length))
            return cache
        except Exception:
            pass

    layers = getattr(cache, "layers", None)
    if layers:
        for layer in layers:
            if getattr(layer, "keys", None) is not None:
                layer.keys = layer.keys[..., :max_length, :]
                layer.values = layer.values[..., :max_length, :]
        return cache

    key_cache = getattr(cache, "key_cache", None)
    if key_cache is not None:
        for i in range(len(key_cache)):
            cache.key_cache[i] = cache.key_cache[i][..., :max_length, :]
            cache.value_cache[i] = cache.value_cache[i][..., :max_length, :]
        return cache

    if isinstance(cache, (tuple, list)):
        return tuple(
            (k[..., :max_length, :], v[..., :max_length, :]) for k, v in cache
        )

    raise TypeError(f"do not know how to crop cache of type {type(cache)!r}")


_SIGNATURE_CACHE: dict[tuple[int, str], bool] = {}


def _accepts(model: torch.nn.Module, param: str) -> bool:
    key = (id(type(model)), param)
    if key not in _SIGNATURE_CACHE:
        try:
            sig = inspect.signature(model.forward)
            _SIGNATURE_CACHE[key] = param in sig.parameters
        except (TypeError, ValueError):
            _SIGNATURE_CACHE[key] = False
    return _SIGNATURE_CACHE[key]


def _logits_kwarg(model: torch.nn.Module) -> Optional[str]:
    """``logits_to_keep`` from transformers 4.49; ``num_logits_to_keep`` before."""
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if _accepts(model, name):
            return name
    return None


def forward_with_cache(
    model: torch.nn.Module,
    full_ids: torch.Tensor,
    cache: Any,
    cached_len: Optional[int] = None,
    logits_to_keep: int = 0,
):
    """Run ``model`` over only the suffix of ``full_ids`` not yet in ``cache``.

    Returns ``(logits, cache, new_cached_len, first_logit_pos)``, where
    ``logits[:, i]`` predicts the token at absolute position
    ``first_logit_pos + i + 1``.

    ``logits_to_keep`` restricts the output head to the last N positions.
    That is a large saving, not a micro-optimisation: the logits tensor is
    ``[1, positions, 151936]`` upcast to fp32, which is 0.58 MiB per position,
    so a 1000-token prefill would otherwise materialise ~580 MB of logits we
    immediately discard -- an order of magnitude more than the KV cache.
    Speculative decoding needs only the last K+1 rows from the target and the
    last row from the draft.
    """
    if cached_len is None:
        cached_len = cache_length(cache)
    full_len = full_ids.shape[1]
    if full_len <= cached_len:
        raise ValueError(
            f"nothing to do: cache already holds {cached_len} of {full_len} tokens"
        )

    new_ids = full_ids[:, cached_len:]
    kwargs: dict[str, Any] = {
        "input_ids": new_ids,
        "past_key_values": cache,
        "use_cache": True,
        "attention_mask": torch.ones(
            full_ids.shape[0], full_len, dtype=torch.long, device=full_ids.device
        ),
    }
    if _accepts(model, "cache_position"):
        kwargs["cache_position"] = torch.arange(
            cached_len, full_len, device=full_ids.device
        )

    first_logit_pos = cached_len
    n_new = full_len - cached_len
    if logits_to_keep and logits_to_keep < n_new:
        name = _logits_kwarg(model)
        if name is not None:
            kwargs[name] = logits_to_keep
            first_logit_pos = full_len - logits_to_keep

    out = model(**kwargs)
    returned = getattr(out, "past_key_values", None)
    if returned is not None:
        cache = returned

    # a model may ignore the hint; trust the tensor, not the request
    actual = full_len - out.logits.shape[1]
    if actual != first_logit_pos:
        first_logit_pos = actual
    return out.logits, cache, full_len, first_logit_pos


@torch.no_grad()
def logits_for_row(model, ids: torch.Tensor, row: int, chunk: int):
    """Logits for absolute ``row``, with the last ``chunk`` tokens fed at once.

    ``chunk=1`` reproduces autoregressive decoding (one token per forward);
    ``chunk=K+1`` reproduces speculative verification (a block per forward).
    The prefix and the weights are identical either way -- only the matmul
    shape differs, which is what makes this the right probe for deciding
    whether a greedy divergence is a bug or just floating-point ordering.
    """
    start = row + 1 - chunk
    if start < 1:
        raise ValueError(
            f"row {row} is too early for chunk {chunk}; need row+1-chunk >= 1"
        )
    cache = new_cache()
    _, cache, cached, _ = forward_with_cache(
        model, ids[:, :start], cache, 0, logits_to_keep=1
    )
    logits, cache, cached, _ = forward_with_cache(
        model, ids[:, : row + 1], cache, cached, logits_to_keep=chunk
    )
    return logits[0, -1].detach().float()
