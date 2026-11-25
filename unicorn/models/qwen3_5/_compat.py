"""Compatibility shims for the vendored Qwen3.5 modeling code.

The upstream ``qwen3_5`` model targets ``transformers >= 5``; this project pins
``transformers ~= 4.57.6``. Almost every building block Qwen3.5 needs (the
``qwen3`` / ``qwen3_next`` / ``qwen3_vl`` layers, caches, masking, rope) already
exists in 4.57. This module re-implements only the small set of
transformers-5-only helpers that ``modeling_qwen3_5.py`` imports and that are
absent from 4.57:

- ``initialization`` (``init.ones_``/``zeros_``/``copy_``)
- ``vision_utils.{get_vision_bilinear_indices_and_weights,get_vision_cu_seqlens,get_vision_position_ids}``
- ``utils.output_capturing.capture_outputs``
- ``utils.torch_compilable_check``
- ``utils.generic.{is_flash_attention_requested,maybe_autocast,accepts_precomputed_kwargs,merge_with_config_defaults}``

The vision helpers are copied verbatim from transformers ``main``; the decorators
are faithful (or, where Qwen3.5 does not exercise their behavior, pass-through)
reimplementations.
"""

from __future__ import annotations

import inspect
import re
import types
from contextlib import nullcontext
from functools import wraps

import torch
import torch.nn.functional as F

__all__ = [
    "init",
    "auto_docstring",
    "capture_outputs",
    "merge_with_config_defaults",
    "accepts_precomputed_kwargs",
    "torch_compilable_check",
    "is_flash_attention_requested",
    "maybe_autocast",
    "get_vision_bilinear_indices_and_weights",
    "get_vision_cu_seqlens",
    "get_vision_position_ids",
]


# ---------------------------------------------------------------------------
# transformers.initialization (subset used by Qwen3_5PreTrainedModel._init_weights)
# ---------------------------------------------------------------------------
def _ones_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return torch.nn.init.ones_(tensor)
    return tensor


def _zeros_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return torch.nn.init.zeros_(tensor)
    return tensor


def _copy_(tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        with torch.no_grad():
            return tensor.copy_(other)
    return tensor


init = types.SimpleNamespace(ones_=_ones_, zeros_=_zeros_, copy_=_copy_)


# ---------------------------------------------------------------------------
# transformers.utils.auto_docstring
# ---------------------------------------------------------------------------
def auto_docstring(obj=None, **kwargs):
    """No-op replacement.

    The 4.57 ``auto_docstring`` cannot parse the PEP-604 ``X | Y`` union
    annotations used pervasively in the vendored (transformers-5) signatures, and
    it only generates documentation, so we drop it. Supports both bare
    ``@auto_docstring`` and parameterized ``@auto_docstring(checkpoint=...)`` use.
    """
    if obj is None:
        return lambda o: o
    return obj


# ---------------------------------------------------------------------------
# transformers.utils.torch_compilable_check
# ---------------------------------------------------------------------------
def torch_compilable_check(condition, message: str = "") -> None:
    """Compile/trace-safe assertion. Skips while tracing; raises otherwise."""
    if torch.jit.is_tracing():
        return
    try:
        ok = bool(condition)
    except Exception:
        return
    if not ok:
        raise ValueError(message)


# ---------------------------------------------------------------------------
# transformers.utils.generic.is_flash_attention_requested
# ---------------------------------------------------------------------------
def is_flash_attention_requested(config=None, requested_attention_implementation=None, version=None) -> bool:
    if config is not None and requested_attention_implementation is not None:
        raise ValueError("Pass either config or requested_attention_implementation, not both.")
    impl = config._attn_implementation if config is not None else requested_attention_implementation
    if impl is None:
        return False
    if version is not None:
        return re.match(r".*flash.*" + str(version), impl) is not None
    return "flash" in impl


# ---------------------------------------------------------------------------
# transformers.utils.generic.maybe_autocast
# ---------------------------------------------------------------------------
def maybe_autocast(device_type, dtype=None, enabled=True, cache_enabled=None):
    if device_type == "meta":
        return nullcontext()
    if torch.is_autocast_enabled(device_type) or enabled:
        return torch.autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    return nullcontext()


# ---------------------------------------------------------------------------
# transformers.utils.output_capturing.capture_outputs
# ---------------------------------------------------------------------------
def capture_outputs(func):
    """Pass-through.

    Qwen3.5 forwards return ``ModelOutput`` objects directly and the embedding
    pipeline only reads ``last_hidden_state`` / ``pooler_output``; the new
    transformers-5 output-recording machinery is not needed here.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# transformers.utils.generic.merge_with_config_defaults
# ---------------------------------------------------------------------------
def merge_with_config_defaults(func):
    """Pass-through.

    Upstream this injects ``use_cache`` / ``vision_feature_*`` config defaults
    into the wrapped method's kwargs. Qwen3.5's vision feature methods do not
    read any of those, so pass-through preserves behavior while avoiding leaking
    e.g. ``use_cache`` into the vision encoder.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# transformers.utils.generic.accepts_precomputed_kwargs (verbatim port)
# ---------------------------------------------------------------------------
_KNOWN_MODALITIES = ("image", "video", "audio")


def accepts_precomputed_kwargs(modality: str):
    prefix = f"{modality}_"
    other_prefixes = tuple(f"{m}_" for m in _KNOWN_MODALITIES if m != modality)

    def decorator(func):
        existing_params = set(inspect.signature(func).parameters)

        @wraps(func)
        def wrapper(*args, **kwargs):
            translated = {}
            for k, v in kwargs.items():
                if k.startswith(other_prefixes):
                    continue
                if k.startswith(prefix) and k not in existing_params:
                    translated[k.removeprefix(prefix)] = v
                else:
                    translated[k] = v
            return func(*args, **translated)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# transformers.vision_utils (subset, copied verbatim from transformers main)
# ---------------------------------------------------------------------------
def get_vision_cu_seqlens(grid_thw: torch.Tensor, kwargs: dict | None = None) -> torch.Tensor:
    if kwargs is not None and (cu_seqlens := kwargs.pop("cu_seqlens", None)) is not None:
        return cu_seqlens
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
    )
    return F.pad(cu_seqlens, (1, 0), value=0)


def get_vision_position_ids(
    grid_thw: torch.Tensor, spatial_merge_size: int | torch.Tensor, kwargs: dict | None = None
) -> torch.Tensor:
    if kwargs is not None and (position_ids := kwargs.pop("position_ids", None)) is not None:
        return position_ids
    device = grid_thw.device
    if isinstance(spatial_merge_size, int):
        spatial_merge_size = torch.tensor([spatial_merge_size], device=device).expand(len(grid_thw))

    position_ids = []
    for (t, h, w), merge_size in zip(grid_thw.tolist(), spatial_merge_size.tolist()):
        t, h, w, merge_size = int(t), int(h), int(w), int(merge_size)
        hpos_ids = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(h // merge_size, merge_size, w // merge_size, merge_size).transpose(1, 2).flatten()

        wpos_ids = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(h // merge_size, merge_size, w // merge_size, merge_size).transpose(1, 2).flatten()
        position_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

    return torch.cat(position_ids, dim=0)


def get_vision_bilinear_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    spatial_merge_size: int,
    kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kwargs is not None:
        bilinear_indices = kwargs.pop("bilinear_indices", None)
        bilinear_weights = kwargs.pop("bilinear_weights", None)
        if bilinear_indices is not None and bilinear_weights is not None:
            return bilinear_indices, bilinear_weights
    side = num_grid_per_side
    merge_size = spatial_merge_size
    device = grid_thw.device

    idx_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]

    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)

        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)

        h_floor = h_grid.int()
        w_floor = w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)

        h_frac = h_grid - h_floor
        w_frac = w_grid - w_floor

        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side

        corner_indices = [
            (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]

        h_idx = torch.arange(h, device=device).view(h // merge_size, merge_size)
        w_idx = torch.arange(w, device=device).view(w // merge_size, merge_size)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten().repeat(t)

        for i in range(4):
            idx_parts[i].append(corner_indices[i][reorder])
            weight_parts[i].append(corner_weights[i][reorder])

    bilinear_indices = torch.stack([torch.cat(p) for p in idx_parts])
    bilinear_weights = torch.stack([torch.cat(p) for p in weight_parts])
    return bilinear_indices, bilinear_weights
