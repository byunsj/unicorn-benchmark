"""LongLLMLingua query-aware prompt compression model wrapper.

Uses the ``llmlingua`` package with a causal LM (e.g. LLaMA-2-7B) to
perform perplexity-based compression.  LongLLMLingua extends LLMLingua
with four innovations for long-context scenarios:

1. **Contrastive perplexity** — measures how much a document reduces
   the perplexity of the question (query-aware compression).
2. **Document reordering** — mitigates "lost in the middle" bias.
3. **Dynamic compression ratios** — allocates more token budget to
   relevant documents.
4. **Subsequence recovery** — restores token integrity after pruning.

Unlike LLMLingua-2 (which is query-agnostic and uses a small encoder),
LongLLMLingua is **query-aware** and uses a full causal LM, making it
more expensive but potentially higher quality for RAG scenarios.

Reference: https://aclanthology.org/2024.acl-long.91/
Paper:     https://arxiv.org/abs/2310.06839
"""

from __future__ import annotations

import functools
import logging

from .base import CompressionModel, build_spans_from_pruned

logger = logging.getLogger(__name__)


def _patch_model_for_legacy_kv_cache(compressor):
    """Work around transformers >= 4.38 expecting ``DynamicCache``.

    The ``llmlingua`` ``iterative_compress_prompt`` method slices and
    reconstructs the KV cache as a plain list of ``[key, value]`` pairs,
    but newer ``transformers`` ``LlamaForCausalLM.forward`` calls
    ``past_key_values.get_seq_length()``, which fails on a list.

    This wraps the inner model's ``forward`` so that any legacy list-of-lists
    KV cache is converted to a ``DynamicCache`` before reaching the model.
    """
    from transformers.cache_utils import DynamicCache

    original_forward = compressor.model.forward

    @functools.wraps(original_forward)
    def _forward(*args, **kwargs):
        pkv = kwargs.get("past_key_values")
        if isinstance(pkv, list):
            kwargs["past_key_values"] = DynamicCache.from_legacy_cache(pkv)
        return original_forward(*args, **kwargs)

    compressor.model.forward = _forward


class LongLLMLinguaModel(CompressionModel):
    """Wraps ``llmlingua.PromptCompressor`` with LongLLMLingua parameters.

    The ``threshold`` parameter maps to ``rate`` — the target fraction
    of tokens to *keep*.  For example ``threshold=0.55`` retains ~55 %
    of the original tokens.
    """

    MODEL_NAME = "NousResearch/Llama-2-7b-hf"
    HAS_SEPARATE_SCORING = True

    FORCE_TOKENS: list[str] = ["\n", ".", "!", "?", ","]

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
            use_llmlingua2=False,
            device_map=device,
        )
        _patch_model_for_legacy_kv_cache(self.compressor)

    def unload(self) -> None:
        import torch

        del self.compressor
        self.compressor = None
        self.device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self.compressor is not None

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
        threshold: float = 0.55,
        compute_spans: bool = True,
        compute_reranking_score: bool = True,
    ) -> dict:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        if not context or not isinstance(context, str) or not context.strip():
            result: dict = {
                "pruned_context": "",
                "reranking_score": -999.0,
                "compression_rate": 0.0,
            }
            if compute_spans:
                result["spans"] = [{"start": 0, "end": len(context or ""), "selected": False}]
            return result

        rate = max(0.01, min(threshold, 1.0))

        compressed = self.compressor.compress_prompt(
            [context],
            question=query or "",
            rate=rate,
            condition_in_question="after_condition",
            reorder_context="sort",
            dynamic_context_compression_ratio=0.3,
            condition_compare=True,
            context_budget="+100",
            rank_method="longllmlingua",
            force_tokens=self.FORCE_TOKENS,
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
            "[LongLLMLingua] single | rate=%.3f  ratio=%s"
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

        score = self.compute_relevance_score(query, context) if compute_reranking_score else 0.0

        out: dict = {
            "pruned_context": pruned_context,
            "reranking_score": score,
            "compression_rate": compression_pct,
        }
        if compute_spans:
            out["spans"] = build_spans_from_pruned(context, pruned_context)
        return out

    # ------------------------------------------------------------------
    # Document-level relevance scoring (r_k from the paper, Eq. 2)
    # ------------------------------------------------------------------

    _RESTRICT_STMT = " We can get the answer to this question in the given documents."

    def compute_relevance_scores(self, query: str, documents: list[str]) -> list[float]:
        """Score documents with LongLLMLingua's native conditional perplexity."""
        return [self.compute_relevance_score(query, document) for document in documents]

    def compute_relevance_score(self, query: str, document: str) -> float:
        """Compute document relevance via conditional perplexity (r_k).

        Measures how much *document* reduces the perplexity of *query*.
        Lower perplexity ⇒ more relevant.  The returned value is negated
        so that higher score = more relevant, matching the descending sort
        the pipeline applies.

        When ``document + query`` exceeds the model's context window, the
        document is truncated from the end so that the query portion is
        always included in the perplexity calculation.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        if not document or not isinstance(document, str) or not document.strip():
            return -999.0

        import torch

        query_text = query + self._RESTRICT_STMT
        max_len = self.compressor.max_position_embeddings
        query_tokens = len(self.compressor.tokenizer.encode(query_text, add_special_tokens=False))
        max_doc_tokens = max_len - query_tokens - 2  # reserve space for special tokens

        if max_doc_tokens <= 0:
            return -999.0

        doc_token_ids = self.compressor.tokenizer.encode(document, add_special_tokens=False)
        if len(doc_token_ids) > max_doc_tokens:
            doc_token_ids = doc_token_ids[:max_doc_tokens]
            document = self.compressor.tokenizer.decode(doc_token_ids)
            logger.debug(
                "[LongLLMLingua] document truncated to %d tokens (max_pos=%d, query=%d)",
                max_doc_tokens,
                max_len,
                query_tokens,
            )

        with torch.inference_mode():
            ppl = self.compressor.get_condition_ppl(
                document,
                query_text,
                condition_in_question="after",
            )
        score = -float(ppl)
        logger.debug(
            "[LongLLMLingua] relevance score | ppl=%.4f  score=%.4f  doc_chars=%d",
            float(ppl),
            score,
            len(document),
        )
        return score

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
        threshold: float = 0.55,
        batch_size: int = 0,
        compute_spans: bool = True,
        compute_reranking_score: bool = True,
    ) -> list[dict]:
        """LongLLMLingua has no native batch API, so we loop."""
        return [
            self.process_with_highlights(
                query,
                doc,
                threshold,
                compute_spans=compute_spans,
                compute_reranking_score=compute_reranking_score,
            )
            for doc in docs
        ]
