"""BGE document reranker with no context pruning.

The rerank-only baseline. It isolates what reranking alone contributes by
selecting documents without shortening them, so ``compression_rate`` is always
zero and the agent sees full document text.
"""

from __future__ import annotations

from typing import Any

from ._crossencoder import DEFAULT_BGE_RERANKER_MODEL
from .base import CompressionModel


class BGERerankerModel(CompressionModel):
    """Score documents with BGE and return their text unchanged."""

    MODEL_NAME = DEFAULT_BGE_RERANKER_MODEL

    def __init__(
        self,
        display_name: str | None = None,
        lazy: bool = True,
        **kwargs,
    ):
        if not kwargs.get("external_reranker_name"):
            kwargs["external_reranker_name"] = self.MODEL_NAME
        super().__init__(**kwargs)
        self.display_name = display_name or "BGE reranker v2-m3 (rerank only)"
        self.device: str | None = None

        if not lazy:
            self.load()

    def load(self, device: str = "auto") -> None:
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.load_external_reranker(device)

    def unload(self) -> None:
        self.unload_external_reranker()
        self.device = None

    def is_loaded(self) -> bool:
        return self.is_external_reranker_loaded()

    def get_info(self) -> dict:
        return {
            "name": self.MODEL_NAME,
            "display_name": self.display_name,
            "loaded": self.is_loaded(),
            "device": str(self.device) if self.device else None,
            **self.get_external_reranker_info(),
        }

    @property
    def tokenizer(self):
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        return self._external_reranker.model.tokenizer

    def process_with_highlights(
        self,
        query: str,
        context: str,
        threshold: float = 0.0,
        compute_spans: bool = True,
    ) -> dict[str, Any]:
        score = self.compute_relevance_score(query, context)
        result: dict[str, Any] = {
            "pruned_context": context,
            "reranking_score": score,
            "compression_rate": 0.0,
        }
        if compute_spans:
            result["spans"] = [{"start": 0, "end": len(context), "selected": True, "score": score}]
        return result
