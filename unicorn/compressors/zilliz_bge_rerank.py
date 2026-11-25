"""Zilliz Semantic Highlight pruning on top of a BGE reranker.

The base Zilliz model has no reranking head, so on its own it cannot select
top-k documents (it prunes whatever it is given). This wrapper uses
``BAAI/bge-reranker-v2-m3`` to score and select the top-k candidates, then
applies Zilliz Semantic Highlight to prune the selected documents.

Exposing ``compute_relevance_score`` puts the pipeline on its two-phase path:
score all candidates with BGE, keep the top-k, prune only those with Zilliz.
"""

from __future__ import annotations

from ._crossencoder import DEFAULT_BGE_RERANKER_MODEL
from .base import CompressionModel
from .zilliz import ZillizSemanticHighlightModel


class ZillizBGERerankModel(CompressionModel):
    BGE_MODEL = DEFAULT_BGE_RERANKER_MODEL
    HAS_SEPARATE_SCORING = True

    def __init__(
        self,
        threshold: float = 0.3,
        display_name: str | None = None,
        lazy: bool = True,
        **kwargs,
    ):
        if not kwargs.get("external_reranker_name"):
            kwargs["external_reranker_name"] = self.BGE_MODEL
        super().__init__(**kwargs)
        self.threshold = threshold
        self.display_name = display_name or "Zilliz Semantic Highlight + BGE rerank"
        self.device = None
        self.zilliz = ZillizSemanticHighlightModel(lazy=lazy)

    def load(self, device: str = "auto") -> None:
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.load_external_reranker(device)
        self.zilliz.load(device=device)

    def unload(self) -> None:
        self.unload_external_reranker()
        self.zilliz.unload()
        self.device = None

    def set_chunk_batch_size(self, value: int | None) -> None:
        """The pruner is a separate object, so bound its encoder too."""
        super().set_chunk_batch_size(value)
        self.zilliz.set_chunk_batch_size(value)

    def is_loaded(self) -> bool:
        return self.is_external_reranker_loaded() and self.zilliz.is_loaded()

    def get_info(self) -> dict:
        return {
            "name": f"{self.BGE_MODEL}+{self.zilliz.MODEL_NAME}",
            "display_name": self.display_name,
            "loaded": self.is_loaded(),
            "device": str(self.device) if self.device else None,
            **self.get_external_reranker_info(),
        }

    @property
    def tokenizer(self):
        return self.zilliz.tokenizer

    def compute_relevance_score(self, query: str, document: str) -> float:
        return self.compute_relevance_scores(query, [document])[0]

    def compute_relevance_scores(self, query: str, documents: list[str]) -> list[float]:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        if not documents:
            return []
        return super().compute_relevance_scores(query, documents)

    def process_with_highlights(
        self,
        query: str,
        context: str,
        threshold: float = 0.5,
        compute_spans: bool = True,
    ) -> dict:
        return self.zilliz.process_with_highlights(query, context, threshold=threshold, compute_spans=compute_spans)

    def process_document(self, query: str, document: str) -> dict:
        return self.process_with_highlights(query, document, compute_spans=False)
