"""Abstract base class for context compression models."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher

DEFAULT_BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

_WORD_RE = re.compile(r"\S+")


def build_spans_from_pruned(original: str, pruned: str) -> list:
    """Build contiguous character-offset spans via LCS word-level alignment.

    Uses :class:`~difflib.SequenceMatcher` to find the longest common
    subsequence between the whitespace-separated words of *original* and
    *pruned*, then marks matched original words as selected.  This is
    robust to tokenizer round-trip artifacts (insertions, deletions,
    substitutions) that would permanently desync a simple two-pointer walk.

    Returns a list of ``{"start": int, "end": int, "selected": bool}``
    dicts that cover *original* with no gaps.
    """
    text_len = len(original)
    if not pruned or not pruned.strip():
        return [{"start": 0, "end": text_len, "selected": False}]

    orig_tokens = list(_WORD_RE.finditer(original))
    pruned_words = pruned.split()

    if not orig_tokens or not pruned_words:
        return [{"start": 0, "end": text_len, "selected": False}]

    orig_words = [m.group() for m in orig_tokens]

    matcher = SequenceMatcher(None, orig_words, pruned_words, autojunk=False)

    matched = [False] * len(orig_words)
    for a_start, _, size in matcher.get_matching_blocks():
        for k in range(size):
            matched[a_start + k] = True

    if not any(matched):
        return [{"start": 0, "end": text_len, "selected": False}]

    selected_ranges: list[tuple[int, int]] = []
    i = 0
    while i < len(matched):
        if matched[i]:
            start_idx = i
            while i < len(matched) and matched[i]:
                i += 1
            char_start = orig_tokens[start_idx].start()
            char_end = orig_tokens[i - 1].end()
            selected_ranges.append((char_start, char_end))
        else:
            i += 1

    spans: list[dict] = []
    pos = 0
    for sel_start, sel_end in selected_ranges:
        if pos < sel_start:
            spans.append({"start": pos, "end": sel_start, "selected": False})
        spans.append({"start": sel_start, "end": sel_end, "selected": True})
        pos = sel_end
    if pos < text_len:
        spans.append({"start": pos, "end": text_len, "selected": False})

    return spans


class ExternalRerankerPlugin:
    """Optional full-document BGE selector shared by compression models.

    Subclasses opt in by accepting the ``external_reranker_*`` constructor
    fields and forwarding them to ``super().__init__``. The plugin owns the
    scorer lifecycle while the compression model retains ownership of its
    native model and compression behavior.
    """

    HAS_SEPARATE_SCORING = False

    def __init__(self, **kwargs):
        external_reranker_name = kwargs.pop("external_reranker_name", None)
        external_reranker_batch_size = kwargs.pop("external_reranker_batch_size", 16)
        external_reranker_max_length = kwargs.pop("external_reranker_max_length", 8192)
        super().__init__(**kwargs)
        self.external_reranker_name = external_reranker_name
        self.external_reranker_batch_size = external_reranker_batch_size
        self.external_reranker_max_length = external_reranker_max_length
        self._external_reranker = None
        self.HAS_SEPARATE_SCORING = external_reranker_name is not None

    #: Rows per encoder forward. ``None`` leaves each checkpoint at its own
    #: default; set it explicitly to bound memory on long documents.
    chunk_batch_size: int | None = None

    def set_chunk_batch_size(self, value: int | None) -> None:
        """Bound every encoder in this model, external scorer included."""
        self.chunk_batch_size = None if value is None else max(1, int(value))
        if self.chunk_batch_size is None:
            return
        self.external_reranker_batch_size = self.chunk_batch_size
        scorer = getattr(self, "_external_reranker", None)
        if scorer is not None:
            scorer.batch_size = self.chunk_batch_size

    def _chunk_batch(self, fallback: int) -> int:
        """Rows per forward: the explicit override if set, else *fallback*."""
        override = self.chunk_batch_size
        return fallback if override is None else max(1, int(override))

    @property
    def has_external_reranker(self) -> bool:
        return self._external_reranker is not None

    def is_external_reranker_loaded(self) -> bool:
        """Return whether the configured external scorer is loaded."""
        return self._external_reranker is not None and self._external_reranker.is_loaded()

    def load_external_reranker(self, device: str) -> None:
        """Load the configured full-document BGE selector, if any."""
        if not self.external_reranker_name:
            return
        from ._crossencoder import create_bge_reranker

        self._external_reranker = create_bge_reranker(
            model_name=self.external_reranker_name,
            batch_size=self.external_reranker_batch_size,
            max_length=self.external_reranker_max_length,
        )
        self._external_reranker.load(device)

    def unload_external_reranker(self) -> None:
        if self._external_reranker is not None:
            self._external_reranker.unload()
            self._external_reranker = None

    def compute_relevance_scores(self, query: str, documents: list[str]) -> list[float]:
        """Score complete documents with the configured external reranker.

        A compressor with native scoring may override this method and delegate
        here when ``has_external_reranker`` is true.
        """
        if self._external_reranker is None:
            raise RuntimeError("No external reranker configured")
        return self._external_reranker.score_docs(query, documents)

    def compute_relevance_score(self, query: str, document: str) -> float:
        return self.compute_relevance_scores(query, [document])[0]

    def get_external_reranker_info(self) -> dict:
        """Return serializable metadata for the optional external selector."""
        return {
            "external_reranker": self.external_reranker_name,
            "external_reranker_loaded": self.is_external_reranker_loaded(),
            "external_reranker_batch_size": self.external_reranker_batch_size,
            "external_reranker_max_length": self.external_reranker_max_length,
        }

    # Backward-compatible alias for subclasses written before the generic
    # ``compute_relevance_scores`` implementation moved to this plugin.
    def compute_external_relevance_scores(self, query: str, documents: list[str]) -> list[float]:
        return self.compute_relevance_scores(query, documents)


class CompressionModel(ExternalRerankerPlugin, ABC):
    """Unified interface for hard context compression models.

    All models must implement :meth:`process_with_highlights`, which
    returns pruned text (and optionally character-offset spans for
    highlighting in a UI).

    Two independent batching levels exist and must not be conflated. The
    ``batch_size`` argument of :meth:`process_batch_with_highlights` groups
    *documents* per call, which only changes how work is handed to the model.
    :attr:`chunk_batch_size` bounds the *encoder* batch, i.e. how many rows
    reach a single model forward, and that is what determines GPU time and
    peak memory. Several checkpoints flatten a call's documents into their own
    block list and forward it 32 or 50 rows at a time no matter how the
    documents were grouped, so only the second knob controls accelerator cost.
    """

    @abstractmethod
    def load(self, device: str = "auto") -> None:
        """Load the model onto the specified device ('auto', 'cpu', 'cuda')."""

    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free memory."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the model is currently loaded."""

    @abstractmethod
    def get_info(self) -> dict:
        """Return model metadata: name, loaded status, device."""

    @abstractmethod
    def process_with_highlights(
        self,
        query: str,
        context: str,
        threshold: float = 0.1,
        compute_spans: bool = True,
    ) -> dict:
        """Process a single query-context pair.

        Returns:
            {
                "pruned_context": str,
                "reranking_score": float,
                "compression_rate": float,
                "spans": [{"start": int, "end": int, "selected": bool}],
                    # only present when *compute_spans* is True
            }

        ``start`` / ``end`` are character offsets into the original *context*.
        Spans are contiguous: span[i].end == span[i+1].start,
        span[0].start == 0, span[-1].end == len(context).
        """

    def process_batch_with_highlights(
        self,
        query: str,
        docs: list[str],
        threshold: float = 0.1,
        batch_size: int = 0,
        compute_spans: bool = True,
    ) -> list[dict]:
        """Process multiple documents for a single query.

        Default implementation calls :meth:`process_with_highlights` in
        a loop.  Subclasses may override for batch-efficient inference.
        """
        return [self.process_with_highlights(query, doc, threshold, compute_spans=compute_spans) for doc in docs]
