"""Zilliz Semantic Highlight (bilingual) context compression model wrapper.

The model does its own internal sentence splitting and returns
``highlighted_sentences`` (the kept sentences) plus per-sentence
``sentence_probabilities``.  Since we cannot access the model's
internal sentence boundaries, we locate each highlighted sentence
in the original text via order-aware substring search and build
contiguous character-offset spans for the highlighting UI.

Reference: https://huggingface.co/zilliz/semantic-highlight-bilingual-v1
"""

from __future__ import annotations

import logging

from .base import CompressionModel

logger = logging.getLogger(__name__)


class ZillizSemanticHighlightModel(CompressionModel):
    MODEL_NAME = "zilliz/semantic-highlight-bilingual-v1"

    def __init__(self, display_name: str | None = None, lazy: bool = True):
        self.display_name = display_name or self.MODEL_NAME
        self.model = None
        self.device = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, device: str = "auto") -> None:
        import torch
        from transformers import AutoModel

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True, low_cpu_mem_usage=False)
        self.model = self.model.to(device)
        self.model.eval()

    def unload(self) -> None:
        import torch

        del self.model
        self.model = None
        self.device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_info(self) -> dict:
        return {
            "name": self.MODEL_NAME,
            "display_name": self.display_name,
            "loaded": self.is_loaded(),
            "device": str(self.device) if self.device else None,
        }

    # ------------------------------------------------------------------
    # Span construction from highlighted sentences
    # ------------------------------------------------------------------

    @staticmethod
    def _build_spans_from_highlighted(text: str, highlighted_sentences: list[str]) -> list:
        """Build contiguous character-offset spans by locating each
        highlighted sentence in the original text via order-aware
        substring search.

        Returns a list of ``{start, end, selected}`` dicts that cover
        the full text with no gaps.
        """
        selected_ranges = []
        search_pos = 0
        for sent in highlighted_sentences:
            sent_stripped = sent.strip()
            if not sent_stripped:
                continue
            idx = text.find(sent_stripped, search_pos)
            if idx >= 0:
                selected_ranges.append((idx, idx + len(sent_stripped)))
                search_pos = idx + len(sent_stripped)

        if not selected_ranges:
            return [{"start": 0, "end": len(text), "selected": False}]

        spans: list[dict] = []
        pos = 0
        for sel_start, sel_end in selected_ranges:
            if pos < sel_start:
                spans.append({"start": pos, "end": sel_start, "selected": False})
            spans.append({"start": sel_start, "end": sel_end, "selected": True})
            pos = sel_end

        if pos < len(text):
            spans.append({"start": pos, "end": len(text), "selected": False})

        return spans

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
        import torch

        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        # The remote code splits the document into ~max_length-token blocks and
        # encodes them 32 at a time, so looping one document per call does not
        # by itself bound the encoder batch. Only override when asked.
        process_kwargs: dict = {"batch_size": self._chunk_batch(1)} if self.chunk_batch_size else {}

        with torch.inference_mode():
            result = self.model.process(
                question=query,
                context=context,
                threshold=threshold,
                return_sentence_metrics=True,
                **process_kwargs,
            )

        highlighted_sentences = result.get("highlighted_sentences", [])
        compression_rate = result.get("compression_rate", 0.0)
        sentence_probs = result.get("sentence_probabilities", [])

        pruned_context = " ".join(highlighted_sentences)

        max_prob_for_log = max(sentence_probs) if sentence_probs else 0.0

        logger.info(
            "[Zilliz] single | threshold=%.3f  max_prob=%.4f  compression=%.1f%%"
            "  orig_chars=%d  pruned_chars=%d  highlighted_sents=%d"
            "  sent_probs=%s",
            threshold,
            max_prob_for_log,
            float(compression_rate),
            len(context),
            len(pruned_context),
            len(highlighted_sentences),
            [round(p, 4) for p in sentence_probs],
        )

        out: dict = {
            "pruned_context": pruned_context,
            "reranking_score": 0.0,
            "compression_rate": float(compression_rate),
        }
        if compute_spans:
            out["spans"] = self._build_spans_from_highlighted(context, highlighted_sentences)
        return out

    def process_batch_with_highlights(
        self,
        query: str,
        docs: list[str],
        threshold: float = 0.5,
        batch_size: int = 0,
        compute_spans: bool = True,
    ) -> list[dict]:
        """The Zilliz model does not have native batch support, so we loop."""
        return [self.process_with_highlights(query, doc, threshold, compute_spans=compute_spans) for doc in docs]

    # ------------------------------------------------------------------
    # Accessors used by the runner
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        """Tokenizer used by the runner to count tokens before and after pruning."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        return self._tokenizer

    def process_document(self, query: str, document: str) -> dict:
        """Process a single document without computing spans.

        Returns ``{"pruned_context", "reranking_score", "compression_rate"}``.
        """
        return self.process_with_highlights(query, document, compute_spans=False)
