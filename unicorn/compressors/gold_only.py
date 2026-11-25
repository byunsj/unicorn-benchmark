"""Backs the ``gold-only`` oracle row.

The class itself does nothing: every document scores 0.0 and comes back
untouched. The oracle lives in the runner, which hands this compressor only the
gold documents. The result is perfect retrieval at the same document budget as
every other method, with nothing pruned — the ceiling that compression is
trying to preserve.

Only a tokenizer is loaded, for the token counts every run reports.
"""

from __future__ import annotations

from typing import Any

from .base import CompressionModel


class GoldOnlyOracle(CompressionModel):
    """Pass-through compressor: returns the input text unchanged with a flat score."""

    def __init__(
        self,
        model_name: str = "gold-only",
        tokenizer_name: str = "Qwen/Qwen3.5-9B",
        display_name: str | None = None,
        lazy: bool = True,
    ):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.display_name = display_name or model_name
        self._tokenizer = None
        self.device = None

        if not lazy:
            self.load()

    def load(self, device: str = "auto") -> None:
        from transformers import AutoTokenizer

        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            trust_remote_code=True,
        )

    def unload(self) -> None:
        self._tokenizer = None
        self.device = None

    def is_loaded(self) -> bool:
        return self._tokenizer is not None

    def get_info(self) -> dict:
        return {
            "name": self.model_name,
            "display_name": self.display_name,
            "loaded": self.is_loaded(),
            "device": str(self.device) if self.device else None,
            "tokenizer": self.tokenizer_name,
        }

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("GoldOnlyOracle not loaded — call load() first.")
        return self._tokenizer

    def process_with_highlights(
        self,
        query: str,
        context: str,
        threshold: float | None = None,
        compute_spans: bool = True,
    ) -> dict:
        text = context if isinstance(context, str) else ""
        result: dict[str, Any] = {
            "pruned_context": text,
            "reranking_score": 0.0,
            "compression_rate": 0.0,
        }
        if compute_spans:
            result["spans"] = [{"start": 0, "end": len(text), "selected": True}]
        return result

    def process_batch_with_highlights(
        self,
        query: str,
        docs: list[str],
        threshold: float | None = None,
        batch_size: int = 0,
        compute_spans: bool = True,
    ) -> list[dict]:
        return [self.process_with_highlights(query, d, threshold, compute_spans=compute_spans) for d in docs]
