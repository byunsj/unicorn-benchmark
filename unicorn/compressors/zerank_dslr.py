"""zerank-DSLR: DSLR-style sentence-level pruning scored with zerank-2.

Each document is split into sentences and scored individually, following DSLR
(https://arxiv.org/abs/2407.03627). NLTK Punkt is the default splitter; spaCy's
``xx_sent_ud_sm`` is available for multilingual segmentation. Raw logits become
calibrated probabilities via ``sigmoid(logit/5)`` (per the model card), and
sentences scoring at or above the threshold are kept in their original order.
No per-document token budget is applied.
"""

from __future__ import annotations

import gc
import importlib

import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

from .base import CompressionModel, build_spans_from_pruned


def _ensure_nltk_punkt() -> None:
    import nltk

    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/", 1)[1], quiet=True)


def _ensure_pad_token(cross_encoder) -> None:
    """zerank-2 is Qwen3ForSequenceClassification, which needs a pad_token_id to
    pool the last non-pad token when batch_size > 1. Some checkpoints leave
    config.pad_token_id unset, so derive it from the tokenizer (falling back to
    eos) to avoid 'Cannot handle batch sizes > 1 if no padding token is defined.'
    """
    tok = getattr(cross_encoder, "tokenizer", None)
    hf_model = getattr(cross_encoder, "model", None)
    if tok is None or hf_model is None:
        return
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    if getattr(hf_model.config, "pad_token_id", None) is None:
        hf_model.config.pad_token_id = tok.pad_token_id


class ZerankDSLR(CompressionModel):
    """Hard context pruning with zerank-2 thresholded probabilities (sentence-level)."""

    DEFAULT_BATCH_SIZE = 16
    DEFAULT_SPACY_MODEL = "xx_sent_ud_sm"
    # Cap pair length so an unsplit giant "sentence" (e.g. Korean text NLTK
    # can't split) doesn't OOM the cross-encoder. Normal sentences are shorter.
    MAX_PAIR_TOKENS = 2048

    def __init__(
        self,
        model_name: str = "zeroentropy/zerank-2",
        threshold: float = 0.4,
        display_name: str | None = None,
        lazy: bool = True,
        sentence_splitter: str = "nltk",
        spacy_model: str = DEFAULT_SPACY_MODEL,
    ):
        if sentence_splitter not in {"nltk", "spacy"}:
            raise ValueError(f"Unsupported sentence_splitter={sentence_splitter!r}; expected 'nltk' or 'spacy'.")
        self.model_name = model_name
        self.display_name = display_name or model_name
        self.threshold = threshold
        self.sentence_splitter = sentence_splitter
        self.spacy_model = spacy_model
        self.model = None
        self._tokenizer = None
        self._nlp = None
        self.device = None

    def load(self, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = CrossEncoder(
            self.model_name,
            device=device,
            trust_remote_code=True,
            max_length=self.MAX_PAIR_TOKENS,
        )
        _ensure_pad_token(self.model)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.sentence_splitter == "nltk":
            _ensure_nltk_punkt()
        else:
            try:
                spacy = importlib.import_module("spacy")
                self._nlp = spacy.load(self.spacy_model)
            except OSError as exc:
                raise RuntimeError(
                    f"spaCy sentence model {self.spacy_model!r} is not installed. "
                    "Install the model declared in eval_package/pyproject.toml."
                ) from exc

    def unload(self) -> None:
        del self.model
        del self._tokenizer
        self.model = None
        self._tokenizer = None
        self._nlp = None
        self.device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self.model is not None and self._tokenizer is not None

    def get_info(self) -> dict:
        return {
            "name": self.model_name,
            "display_name": self.display_name,
            "loaded": self.is_loaded(),
            "device": str(self.device) if self.device else None,
            "granularity": "sentence",
            "sentence_splitter": self.sentence_splitter,
            "spacy_model": self.spacy_model if self.sentence_splitter == "spacy" else None,
        }

    @property
    def tokenizer(self):
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        return self._tokenizer

    def _prob_from_raw_score(self, raw_score: float) -> float:
        score_tensor = torch.tensor(raw_score, dtype=torch.float32)
        return float(torch.sigmoid(score_tensor / 5.0).item())

    def _predict_raw_scores(self, pairs: list[tuple[str, str]], batch_size: int):
        """Return unactivated logits so calibration is applied exactly once."""
        return self.model.predict(
            pairs,
            batch_size=batch_size,
            activation_fn=torch.nn.Identity(),
        )

    def _split_into_sentences(self, text: str) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            return []

        if self.sentence_splitter == "spacy":
            if self._nlp is None:
                raise RuntimeError("spaCy sentence splitter is not loaded")
            return [sent.text for sent in self._nlp(text).sents if sent.text and sent.text.strip()]

        from nltk.tokenize import sent_tokenize

        return [s for s in sent_tokenize(text) if s and s.strip()]

    @staticmethod
    def _select_sentences(
        sentences: list[str],
        sentence_probs: list[float],
        threshold: float,
    ) -> list[str]:
        return [s for s, p in zip(sentences, sentence_probs) if p >= threshold]

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
            result: dict = {
                "pruned_context": "",
                "reranking_score": -999.0,
                "compression_rate": 0.0,
            }
            if compute_spans:
                result["spans"] = [{"start": 0, "end": len(context or ""), "selected": False}]
            return result

        th = threshold if threshold is not None else self.threshold

        sentences = self._split_into_sentences(context) or [context]
        pairs = [(query, s) for s in sentences]
        raw_scores = self._predict_raw_scores(pairs, batch_size=self._chunk_batch(self.DEFAULT_BATCH_SIZE))
        sentence_probs = [self._prob_from_raw_score(float(s)) for s in raw_scores]

        kept_sentences = self._select_sentences(sentences, sentence_probs, th)
        pruned_context = "\n\n".join(kept_sentences) if kept_sentences else ""
        doc_score = max(sentence_probs) if sentence_probs else 0.0

        orig_len = len(context)
        compression_rate = 100.0 * (1.0 - len(pruned_context) / orig_len) if orig_len else 0.0

        out: dict = {
            "pruned_context": pruned_context,
            "reranking_score": doc_score,
            "compression_rate": compression_rate,
        }
        if compute_spans:
            out["spans"] = build_spans_from_pruned(context, pruned_context)
        return out

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
        if not docs:
            return []

        th = threshold if threshold is not None else self.threshold
        eff_batch_size = self._chunk_batch(batch_size if batch_size > 0 else self.DEFAULT_BATCH_SIZE)

        per_doc_sentences: list[list[str]] = []
        flat_pairs: list[tuple[str, str]] = []
        flat_doc_idx: list[int] = []
        for di, doc in enumerate(docs):
            text = doc if isinstance(doc, str) else ""
            if not text.strip():
                per_doc_sentences.append([])
                continue
            sentences = self._split_into_sentences(text) or [text]
            per_doc_sentences.append(sentences)
            for s in sentences:
                flat_pairs.append((query, s))
                flat_doc_idx.append(di)

        if flat_pairs:
            flat_raw = self._predict_raw_scores(flat_pairs, batch_size=eff_batch_size)
            flat_probs = [self._prob_from_raw_score(float(s)) for s in flat_raw]
        else:
            flat_probs = []

        per_doc_probs: list[list[float]] = [[] for _ in docs]
        for di, p in zip(flat_doc_idx, flat_probs):
            per_doc_probs[di].append(p)

        results: list[dict] = []
        for doc, sentences, sentence_probs in zip(docs, per_doc_sentences, per_doc_probs):
            text = doc if isinstance(doc, str) else ""
            if not text.strip() or not sentences:
                item: dict = {
                    "pruned_context": "",
                    "reranking_score": -999.0,
                    "compression_rate": 0.0,
                }
                if compute_spans:
                    item["spans"] = [{"start": 0, "end": len(text), "selected": False}]
                results.append(item)
                continue

            kept_sentences = self._select_sentences(sentences, sentence_probs, th)
            pruned_context = "\n\n".join(kept_sentences) if kept_sentences else ""
            doc_score = max(sentence_probs) if sentence_probs else 0.0
            comp = 100.0 * (1.0 - len(pruned_context) / len(text)) if len(text) else 0.0

            item = {
                "pruned_context": pruned_context,
                "reranking_score": doc_score,
                "compression_rate": comp,
            }
            if compute_spans:
                item["spans"] = build_spans_from_pruned(text, pruned_context)
            results.append(item)

        return results
