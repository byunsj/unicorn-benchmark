"""LLMLingua-2 task-agnostic prompt compression model wrapper.

Uses the ``llmlingua`` package to load an XLM-RoBERTa-large model
fine-tuned for per-token binary classification (preserve / discard).
Compression is achieved by retaining the top-rated tokens ranked by
preservation probability.

**Important:** LLMLingua-2 is *query-agnostic* — the ``query``
parameter accepted by :meth:`process_with_highlights` is ignored.
This is a fundamental difference from Provence / XProvence / Zilliz,
which are all query-aware.

Reference: https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank
Paper:     https://arxiv.org/abs/2403.12968
"""

from __future__ import annotations

import logging

from .base import CompressionModel, build_spans_from_pruned

logger = logging.getLogger(__name__)


class LLMLingua2Model(CompressionModel):
    """Wraps ``llmlingua.PromptCompressor`` for token-level hard compression.

    The ``threshold`` parameter maps to LLMLingua-2's ``rate`` — the
    target fraction of tokens to *keep*.  For example ``threshold=0.5``
    retains ~50 % of the original tokens.
    """

    MODEL_NAME = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"

    FORCE_TOKENS: list[str] = ["\n", ".", "!", "?", ","]
    CHUNK_END_TOKENS: list[str] = [".", "\n"]

    def __init__(
        self,
        model_name: str | None = None,
        display_name: str | None = None,
        lazy: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name or self.MODEL_NAME
        self.display_name = display_name or self.model_name
        self.compressor = None
        self.device: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, device: str = "auto") -> None:
        import torch
        from llmlingua import PromptCompressor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.compressor = PromptCompressor(
            model_name=self.model_name,
            use_llmlingua2=True,
            device_map=device,
        )
        if self.external_reranker_name:
            self.load_external_reranker(device)

    def unload(self) -> None:
        import torch

        self.unload_external_reranker()
        del self.compressor
        self.compressor = None
        self.device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self.compressor is not None and (
            not self.external_reranker_name or (self.has_external_reranker and self._external_reranker.is_loaded())
        )

    def get_info(self) -> dict:
        return {
            "name": self.model_name,
            "display_name": self.display_name,
            "loaded": self.is_loaded(),
            "device": str(self.device) if self.device else None,
            "external_reranker": self.external_reranker_name,
        }

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_with_highlights(
        self,
        query: str,
        context: str,
        threshold: float = 0.5,
        compute_spans: bool = True,
    ) -> dict:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        if not context or not isinstance(context, str) or not context.strip():
            result: dict = {
                "pruned_context": "",
                "reranking_score": 0.0,
                "compression_rate": 0.0,
            }
            if compute_spans:
                result["spans"] = [{"start": 0, "end": len(context or ""), "selected": False}]
            return result

        rate = max(0.01, min(threshold, 1.0))

        compressed = self.compressor.compress_prompt_llmlingua2(
            context,
            rate=rate,
            force_tokens=self.FORCE_TOKENS,
            chunk_end_tokens=self.CHUNK_END_TOKENS,
            return_word_label=True,
            drop_consecutive=True,
        )

        pruned_context = compressed["compressed_prompt"]
        origin_tokens = compressed.get("origin_tokens", 0)
        compressed_tokens = compressed.get("compressed_tokens", 0)

        if origin_tokens > 0:
            compression_pct = (1.0 - compressed_tokens / origin_tokens) * 100.0
        else:
            compression_pct = 0.0

        logger.info(
            "[LLMLingua-2] single | rate=%.3f  ratio=%s"
            "  compression=%.1f%%  orig_tokens=%d  compressed_tokens=%d"
            "  orig_chars=%d  pruned_chars=%d",
            rate,
            compressed.get("ratio", "?"),
            compression_pct,
            origin_tokens,
            compressed_tokens,
            len(context),
            len(pruned_context),
        )

        out: dict = {
            "pruned_context": pruned_context,
            "reranking_score": 0.0,
            "compression_rate": compression_pct,
        }
        if compute_spans:
            out["spans"] = build_spans_from_pruned(context, pruned_context)
        return out

    # ------------------------------------------------------------------
    # Accessors used by the runner
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        """Expose the underlying HuggingFace tokenizer for token counting."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        return self.compressor.tokenizer

    def process_document(self, query: str, document: str) -> dict:
        """Process a single document without computing spans.

        Returns ``{"pruned_context", "reranking_score", "compression_rate"}``.
        """
        return self.process_with_highlights(query, document, compute_spans=False)

    def process_batch_with_highlights(
        self,
        query: str,
        docs: list[str],
        threshold: float = 0.5,
        batch_size: int = 0,
        compute_spans: bool = True,
    ) -> list[dict]:
        """LLMLingua-2 has no native batch API, so we loop."""
        if self.chunk_batch_size is not None and self.is_loaded():
            # The library feeds a document's 512-token chunks through a
            # DataLoader at ``max_batch_size`` (default 50), so looping one
            # document at a time does not by itself bound the encoder batch.
            self.compressor.max_batch_size = self._chunk_batch(1)
        return [self.process_with_highlights(query, doc, threshold, compute_spans=compute_spans) for doc in docs]
