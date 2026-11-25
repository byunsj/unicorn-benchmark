"""Loadable full-document external reranker."""

from __future__ import annotations

import gc

import torch

from .base import DEFAULT_BGE_RERANKER_MODEL


class ExternalReranker:
    """Sentence Transformers cross-encoder that scores complete documents."""

    def __init__(
        self,
        model_name: str = DEFAULT_BGE_RERANKER_MODEL,
        batch_size: int = 16,
        max_length: int = 8192,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.device: str | None = None

    def load(self, device: str = "auto") -> None:
        from sentence_transformers import CrossEncoder

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = CrossEncoder(
            self.model_name,
            device=device,
            max_length=self.max_length,
            trust_remote_code=True,
        )

    def unload(self) -> None:
        self.model = None
        self.device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self.model is not None

    def score_docs(self, query: str, documents: list[str]) -> list[float]:
        if not self.is_loaded():
            raise RuntimeError("External reranker not loaded")
        if not documents:
            return []
        pairs = [(query, document if isinstance(document, str) else "") for document in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        return [float(score) for score in scores]


def create_bge_reranker(
    model_name: str = DEFAULT_BGE_RERANKER_MODEL,
    batch_size: int = 16,
    max_length: int = 8192,
) -> ExternalReranker:
    """Create the shared full-document BGE reranker."""
    return ExternalReranker(model_name=model_name, batch_size=batch_size, max_length=max_length)
