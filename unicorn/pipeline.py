"""Rerank-and-compress step shared by all three tiers.

Compressors differ in when they produce a relevance score, so selection follows
one of three strategies, chosen from the compressor's capabilities:

  retriever  - keep the incoming order, compress only the top-k
  two_phase  - score the pool with a standalone scorer, then compress the top-k
  joint      - score and prune in one pass, rank by the compressor's own scores

OSCAR fits none of them: it compresses and answers in a single call, so it takes
the separate :func:`rerank_and_generate` path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class Compressor(Protocol):
    def process_batch_with_highlights(
        self, query: str, docs: Sequence[str], threshold: float, compute_spans: bool, **kwargs: Any
    ) -> list[dict]: ...


@dataclass
class Candidate:
    doc_id: str
    text: str
    score: float | None = None


@dataclass
class Selected:
    doc_id: str
    text: str
    original_text: str
    score: float | None
    index: int

    @property
    def was_emptied(self) -> bool:
        return not self.text.strip()


@dataclass
class PipelineResult:
    selected: list[Selected]
    ranked_indices: list[int]
    # Aligned to the input pool; None where the strategy never scored a candidate.
    scores: list[float | None]
    strategy: str


@dataclass
class GenerationResult:
    """What an end-to-end compressor produced: an answer, not a compressed context.

    ``memory_vectors`` is how much the decoder was given, which is the only
    measure of compression available when nothing readable is emitted. Each
    :class:`Selected` carries the text that actually reached the compressor, so a
    document truncated to its leading windows is recorded as truncated.
    """

    selected: list[Selected]
    ranked_indices: list[int]
    scores: list[float | None]
    answer: str
    memory_vectors: int
    strategy: str = "end_to_end"


def _rank_descending(scores: Sequence[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)


def _compress(
    compressor: Compressor,
    query: str,
    texts: Sequence[str],
    threshold: float,
    compute_spans: bool,
    batch_kwargs: dict,
    prune: bool,
) -> list[dict]:
    if not prune:
        return [{} for _ in texts]
    return compressor.process_batch_with_highlights(
        query, list(texts), threshold=threshold, compute_spans=compute_spans, **batch_kwargs
    )


def rerank_and_compress(
    compressor: Compressor,
    query: str,
    candidates: Sequence[Candidate],
    top_k: int,
    *,
    threshold: float = 0.1,
    rank_by: str = "compressor",
    prune: bool = True,
    compute_spans: bool = False,
    batch_kwargs: dict[str, Any] | None = None,
) -> PipelineResult:
    """Reduce *candidates* to at most *top_k* compressed documents.

    rank_by="retriever" freezes the incoming order instead of letting the
    compressor reorder. prune=False returns full text but still reranks, which
    is the no-compression baseline.
    """
    if not candidates:
        return PipelineResult([], [], [], "empty")

    batch_kwargs = dict(batch_kwargs or {})
    top_k = min(top_k, len(candidates))
    texts = [c.text for c in candidates]

    score_many = getattr(compressor, "compute_relevance_scores", None)
    score_one = getattr(compressor, "compute_relevance_score", None)
    has_separate_scoring = bool(getattr(compressor, "HAS_SEPARATE_SCORING", False))

    if rank_by == "retriever":
        strategy = "retriever"
        ranked_indices = list(range(len(candidates)))
        top_indices = ranked_indices[:top_k]
        scores: list[float | None] = [c.score for c in candidates]
        outputs = _compress(
            compressor, query, [texts[i] for i in top_indices], threshold, compute_spans, batch_kwargs, prune
        )

    elif has_separate_scoring and (callable(score_many) or callable(score_one)):
        strategy = "two_phase"
        raw = score_many(query, texts) if callable(score_many) else [score_one(query, t) for t in texts]
        phase_scores = [float(s) for s in raw]
        if len(phase_scores) != len(candidates):
            raise ValueError(
                f"{type(compressor).__name__} scored {len(phase_scores)} of {len(candidates)} candidates"
            )
        ranked_indices = _rank_descending(phase_scores)
        top_indices = ranked_indices[:top_k]
        scores = list(phase_scores)
        outputs = _compress(
            compressor, query, [texts[i] for i in top_indices], threshold, compute_spans, batch_kwargs, prune
        )

    else:
        strategy = "joint"
        outputs_all = compressor.process_batch_with_highlights(
            query, texts, threshold=threshold, compute_spans=compute_spans, **batch_kwargs
        )
        joint_scores = [o.get("reranking_score") for o in outputs_all]
        if all(s is not None for s in joint_scores):
            ranked_indices = _rank_descending([float(s) for s in joint_scores])
        else:
            # Query-agnostic pruners emit no score, so keep the retriever's order.
            ranked_indices = list(range(len(candidates)))
        top_indices = ranked_indices[:top_k]
        scores = [None if s is None else float(s) for s in joint_scores]
        outputs = [outputs_all[i] for i in top_indices]

    selected = [
        Selected(
            doc_id=candidates[i].doc_id,
            text=(candidates[i].text if not prune else (out.get("pruned_context") or "").strip()),
            original_text=candidates[i].text,
            score=scores[i],
            index=i,
        )
        for i, out in zip(top_indices, outputs)
    ]
    return PipelineResult(selected, ranked_indices, scores, strategy)


def rerank_and_generate(
    compressor,
    query: str,
    candidates: Sequence[Candidate],
    top_k: int,
    *,
    max_new_tokens: int = 256,
) -> GenerationResult:
    """Select top-k documents and let the compressor's own decoder answer them.

    For OSCAR, which compresses the selected documents into memory embeddings its
    decoder reads directly: there is no intermediate text for a separate generator
    to consume, so selection, compression and answering happen in one call.
    """
    if not candidates:
        return GenerationResult([], [], [], "", 0)

    top_k = min(top_k, len(candidates))
    output = compressor.rerank_and_generate(
        query, [c.text for c in candidates], top_k, max_new_tokens=max_new_tokens
    )
    scores = [float(s) for s in output["scores"]]
    chunks_by_doc = output["selected_chunks_by_doc"]
    selected = [
        Selected(
            doc_id=candidates[i].doc_id,
            text="\n".join(chunks),
            original_text=candidates[i].text,
            score=scores[i],
            index=i,
        )
        for i, chunks in zip(output["top_k_indices"], chunks_by_doc)
    ]
    memory_vectors = sum(len(chunks) for chunks in chunks_by_doc) * output["n_mem_tokens"]
    return GenerationResult(
        selected, output["ranked_indices"], list(scores), output["generated_answer"], memory_vectors
    )
