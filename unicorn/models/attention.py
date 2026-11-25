"""Attention backend selection shared by the vendored architectures."""

from __future__ import annotations

import importlib.util

import torch


def select_attn_implementation(requested: str | None = None) -> str:
    """Resolve the attention backend to load a checkpoint with.

    FlashAttention-2 is what the reported numbers were produced with, but it is an
    optional dependency, so a request for it is downgraded to the universally
    supported path when the package or a GPU is missing. Checkpoint configs carry
    ``flash_attention_2`` themselves, hence the downgrade applies to a caller's
    value too instead of only to the default.
    """
    if requested is not None and requested != "flash_attention_2":
        return requested
    if torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "eager"
