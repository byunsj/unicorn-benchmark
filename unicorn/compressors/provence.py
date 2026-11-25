"""Unified Provence / XProvence reranker with context pruning.

Works with any model whose ``trust_remote_code`` implementation
exposes a ``model.process(question=..., context=..., ...)`` API
(Provence, XProvence, and compatible checkpoints).

Supports two construction modes:

* **Eager** (default, backwards-compatible with the eval pipeline)::

      reranker = ProvenceReranker(model_name=..., threshold=0.1)
      # model is loaded immediately

* **Lazy** (for the webapp / registry)::

      reranker = ProvenceReranker(model_name=..., threshold=0.1, lazy=True)
      reranker.load(device="auto")
"""

from __future__ import annotations

import logging
from typing import Any

from .base import CompressionModel, build_spans_from_pruned

logger = logging.getLogger(__name__)


class ProvenceReranker(CompressionModel):
    """Wraps ``model.process()`` for sentence-level hard compression."""

    SENTENCE_SPLITTER_MAX_CHARS = 8_000_000

    def __init__(
        self,
        model_name: str = "naver/provence-reranker-debertav3-v1",
        threshold: float = 0.5,
        device: str | None = None,
        display_name: str | None = None,
        lazy: bool = False,
        zero_score_when_empty: bool = True,
    ):
        self.model_name = model_name
        self.display_name = display_name or model_name
        self.threshold = threshold
        self.zero_score_when_empty = zero_score_when_empty
        self.model = None
        self.tokenizer = None
        self.device = None

        if not lazy:
            self.load(device=device or "auto")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, device: str = "auto") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, low_cpu_mem_usage=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self._relax_remote_sentence_splitter()

    def _relax_remote_sentence_splitter(self) -> None:
        """Lift spaCy's input-length guard in remote sentence-splitting code.

        XProvence's remote code segments whole documents with spaCy, which
        refuses inputs longer than ``nlp.max_length`` (1M characters by
        default) instead of segmenting them. The guard is a memory precaution
        rather than part of the model, so raising it keeps segmentation and
        pruning identical while allowing full-length corpus documents.
        """
        import sys

        module = sys.modules.get(type(self.model).__module__)
        splitter = getattr(module, "nlp", None)
        current = getattr(splitter, "max_length", None)
        if isinstance(current, int) and current < self.SENTENCE_SPLITTER_MAX_CHARS:
            splitter.max_length = self.SENTENCE_SPLITTER_MAX_CHARS
            logger.info(
                "[%s] raised remote spaCy sentence-splitter limit %d -> %d characters",
                self.model_name.split("/")[-1],
                current,
                self.SENTENCE_SPLITTER_MAX_CHARS,
            )

    def unload(self) -> None:
        import torch

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_info(self) -> dict:
        return {
            "name": self.model_name,
            "display_name": self.display_name,
            "loaded": self.is_loaded(),
            "device": str(self.device) if self.device else None,
        }

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_with_highlights(
        self,
        query: str,
        context: str,
        threshold: float | None = None,
        compute_spans: bool = True,
    ) -> dict:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        if not context or not isinstance(context, str) or not context.strip():
            result: dict[str, Any] = {
                "pruned_context": "",
                "reranking_score": -999.0,
                "compression_rate": 0.0,
            }
            if compute_spans:
                result["spans"] = [{"start": 0, "end": len(context or ""), "selected": False}]
            return result

        threshold = threshold if threshold is not None else self.threshold

        import torch

        extra_kwargs: dict = {}
        if self._is_openprovence():
            extra_kwargs["zero_score_when_empty"] = self.zero_score_when_empty

        with torch.inference_mode():
            out = self.model.process(
                question=query,
                context=context,
                threshold=threshold,
                always_select_title=False,
                enable_warnings=False,
                **extra_kwargs,
            )

        pruned = out["pruned_context"]
        score = float(out["reranking_score"])
        comp = float(out["compression_rate"])

        logger.info(
            "[%s] single | threshold=%.3f  score=%.4f  compression=%.1f%%" "  orig_chars=%d  pruned_chars=%d",
            self.model_name.split("/")[-1],
            threshold,
            score,
            comp,
            len(context),
            len(pruned),
        )

        result = {
            "pruned_context": pruned,
            "reranking_score": score,
            "compression_rate": comp,
        }
        if compute_spans:
            result["spans"] = build_spans_from_pruned(context, pruned)
        return result

    def process_batch_with_highlights(
        self,
        query: str,
        docs: list[str],
        threshold: float | None = None,
        batch_size: int = 0,
        compute_spans: bool = True,
    ) -> list[dict]:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        threshold = threshold if threshold is not None else self.threshold

        if not docs:
            return []

        if batch_size <= 0:
            batch_size = len(docs)

        # ``batch_size`` groups documents per ``process()`` call. The encoder
        # batch is a separate matter: the remote code flattens every document
        # in the call into one list of ~max_len-token blocks and forwards them
        # 32 at a time, so bounding documents leaves the encoder unbounded.
        # Only override when asked, so accuracy runs keep checkpoint defaults.
        process_kwargs: dict = {"batch_size": self._chunk_batch(1)} if self.chunk_batch_size else {}

        import torch

        results: list[dict] = []
        for start in range(0, len(docs), batch_size):
            end = min(start + batch_size, len(docs))
            batch_docs = docs[start:end]

            with torch.inference_mode():
                # OpenProvence expects (question=str, context=List[str]).
                # Naver Provence/XProvence return grouped outputs for
                # (question=[str], context=[List[str]]).
                if self._is_openprovence():
                    out = self.model.process(
                        question=query,
                        context=batch_docs,
                        threshold=threshold,
                        always_select_title=False,
                        enable_warnings=False,
                        zero_score_when_empty=self.zero_score_when_empty,
                        **process_kwargs,
                    )
                else:
                    out = self.model.process(
                        question=[query],
                        context=[batch_docs],
                        threshold=threshold,
                        always_select_title=False,
                        enable_warnings=False,
                        **process_kwargs,
                    )

            pruned_list, score_list, comp_list = self._normalize_batch_outputs(
                out,
                expected_docs=len(batch_docs),
            )

            # Fallback for any unexpected remote-code output shape.
            if not (
                len(pruned_list) == len(batch_docs)
                and len(score_list) == len(batch_docs)
                and len(comp_list) == len(batch_docs)
            ):
                logger.warning(
                    "[%s] unexpected batch output shape (pruned=%d score=%d comp=%d expected=%d); "
                    "falling back to single-doc calls",
                    self.model_name.split("/")[-1],
                    len(pruned_list),
                    len(score_list),
                    len(comp_list),
                    len(batch_docs),
                )
                pruned_list = []
                score_list = []
                comp_list = []
                for doc in batch_docs:
                    with torch.inference_mode():
                        single_out = self.model.process(
                            question=query,
                            context=doc,
                            threshold=threshold,
                            always_select_title=False,
                            enable_warnings=False,
                        )
                    pruned_list.append(single_out["pruned_context"])
                    score_list.append(float(single_out["reranking_score"]))
                    comp_list.append(float(single_out["compression_rate"]))

            for idx, doc in enumerate(batch_docs):
                pruned = str(pruned_list[idx]) if idx < len(pruned_list) else ""
                score = float(score_list[idx]) if idx < len(score_list) else -999.0
                comp = float(comp_list[idx]) if idx < len(comp_list) else 0.0
                logger.info(
                    "[%s] batch doc %d/%d | threshold=%.3f  score=%.4f"
                    "  compression=%.1f%%  orig_chars=%d  pruned_chars=%d",
                    self.model_name.split("/")[-1],
                    start + idx + 1,
                    len(docs),
                    threshold,
                    score,
                    comp,
                    len(doc),
                    len(pruned),
                )
                entry: dict[str, Any] = {
                    "pruned_context": pruned,
                    "reranking_score": score,
                    "compression_rate": comp,
                }
                if compute_spans:
                    entry["spans"] = build_spans_from_pruned(doc, pruned)
                results.append(entry)
        return results

    def _is_openprovence(self) -> bool:
        name = self.model_name.lower()
        return "open-provence" in name or "open_provence" in name

    @staticmethod
    def _normalize_batch_outputs(out: dict[str, Any], expected_docs: int) -> tuple[list, list, list]:
        def _to_list(value: Any) -> list:
            if isinstance(value, (list, tuple)):
                return list(value)
            return [value]

        def _flatten_if_nested_singletons(values: list) -> list:
            flattened = []
            for item in values:
                if isinstance(item, (list, tuple)) and len(item) == 1:
                    flattened.append(item[0])
                else:
                    flattened.append(item)
            return flattened

        pruned_list = _to_list(out.get("pruned_context", []))
        score_list = _to_list(out.get("reranking_score", []))
        comp_list = _to_list(out.get("compression_rate", []))

        # Grouped-by-question shape: [[doc1, doc2, ...]]
        if len(score_list) == 1 and isinstance(score_list[0], (list, tuple)):
            score_list = list(score_list[0])
            if len(pruned_list) == 1 and isinstance(pruned_list[0], (list, tuple)):
                pruned_list = list(pruned_list[0])
            if len(comp_list) == 1 and isinstance(comp_list[0], (list, tuple)):
                comp_list = list(comp_list[0])
            return pruned_list, score_list, comp_list

        # Per-doc nested singleton shape: [[x1], [x2], ...]
        if len(score_list) == expected_docs:
            score_list = _flatten_if_nested_singletons(score_list)
            if len(pruned_list) == expected_docs:
                pruned_list = _flatten_if_nested_singletons(pruned_list)
            if len(comp_list) == expected_docs:
                comp_list = _flatten_if_nested_singletons(comp_list)

        return pruned_list, score_list, comp_list

    # ------------------------------------------------------------------
    # Single-document convenience wrappers
    # ------------------------------------------------------------------

    def process_document(self, query: str, document: str) -> dict[str, Any]:
        """Process a single document without computing spans.

        Returns ``{"pruned_context", "reranking_score", "compression_rate"}``.
        """
        return self.process_with_highlights(query, document, compute_spans=False)
