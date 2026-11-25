"""The three execution strategies must agree on selection semantics."""

import pytest

from unicorn.pipeline import Candidate, rerank_and_compress


class JointCompressor:
    """Scores and prunes in one pass, like Provence."""

    HAS_SEPARATE_SCORING = False

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def process_batch_with_highlights(self, query, docs, threshold, compute_spans, **kwargs):
        self.calls.append(list(docs))
        return [
            {"pruned_context": f"kept:{d}", "reranking_score": self.scores[d]}
            for d in docs
        ]


class TwoPhaseCompressor:
    """Exposes a standalone scorer, like a compressor with an external BGE selector."""

    HAS_SEPARATE_SCORING = True

    def __init__(self, scores):
        self.scores = scores
        self.scored = []
        self.compressed = []

    def compute_relevance_scores(self, query, docs):
        self.scored.append(list(docs))
        return [self.scores[d] for d in docs]

    def process_batch_with_highlights(self, query, docs, threshold, compute_spans, **kwargs):
        self.compressed.append(list(docs))
        return [{"pruned_context": f"kept:{d}"} for d in docs]


class UnscoredCompressor:
    """Query-agnostic pruner that returns no ranking score, like LLMLingua-2."""

    HAS_SEPARATE_SCORING = False

    def process_batch_with_highlights(self, query, docs, threshold, compute_spans, **kwargs):
        return [{"pruned_context": f"kept:{d}", "reranking_score": None} for d in docs]


def pool(*names):
    # Descending retriever scores, so index order is also retriever order.
    return [Candidate(doc_id=n, text=n, score=1.0 - i * 0.1) for i, n in enumerate(names)]


def test_joint_ranks_by_compressor_score():
    scores = {"a": 0.1, "b": 0.9, "c": 0.5}
    compressor = JointCompressor(scores)
    result = rerank_and_compress(compressor, "q", pool("a", "b", "c"), top_k=2)

    assert result.strategy == "joint"
    assert [s.doc_id for s in result.selected] == ["b", "c"]
    assert [s.text for s in result.selected] == ["kept:b", "kept:c"]
    # A joint compressor must see the whole pool, since its scores drive selection.
    assert compressor.calls == [["a", "b", "c"]]


def test_two_phase_compresses_only_the_selected():
    scores = {"a": 0.1, "b": 0.9, "c": 0.5}
    compressor = TwoPhaseCompressor(scores)
    result = rerank_and_compress(compressor, "q", pool("a", "b", "c"), top_k=2)

    assert result.strategy == "two_phase"
    assert [s.doc_id for s in result.selected] == ["b", "c"]
    assert compressor.scored == [["a", "b", "c"]]
    # The point of the two-phase path: never pay compression on unselected docs.
    assert compressor.compressed == [["b", "c"]]


def test_retriever_mode_preserves_order_and_touches_only_top_k():
    compressor = JointCompressor({"a": 0.1, "b": 0.9, "c": 0.5})
    result = rerank_and_compress(
        compressor, "q", pool("a", "b", "c"), top_k=2, rank_by="retriever"
    )

    assert result.strategy == "retriever"
    # Compressor scores are ignored; the retriever's order stands.
    assert [s.doc_id for s in result.selected] == ["a", "b"]
    assert compressor.calls == [["a", "b"]]


def test_unscored_compressor_falls_back_to_retriever_order():
    result = rerank_and_compress(UnscoredCompressor(), "q", pool("a", "b", "c"), top_k=2)

    assert result.strategy == "joint"
    assert [s.doc_id for s in result.selected] == ["a", "b"]
    assert result.scores == [None, None, None]


def test_no_pruning_returns_full_text_but_still_reranks():
    compressor = JointCompressor({"a": 0.1, "b": 0.9, "c": 0.5})
    result = rerank_and_compress(compressor, "q", pool("a", "b", "c"), top_k=2, prune=False)

    assert [s.doc_id for s in result.selected] == ["b", "c"]
    assert [s.text for s in result.selected] == ["b", "c"]


def test_top_k_is_clamped_to_pool_size():
    result = rerank_and_compress(JointCompressor({"a": 0.5}), "q", pool("a"), top_k=5)
    assert len(result.selected) == 1


def test_empty_pool_is_not_an_error():
    result = rerank_and_compress(JointCompressor({}), "q", [], top_k=5)
    assert result.selected == []
    assert result.strategy == "empty"


def test_scorer_returning_wrong_length_is_rejected():
    class Broken(TwoPhaseCompressor):
        def compute_relevance_scores(self, query, docs):
            return [0.1]

    with pytest.raises(ValueError, match="scored 1 of 3"):
        rerank_and_compress(Broken({}), "q", pool("a", "b", "c"), top_k=2)


def test_emptied_documents_are_flagged():
    class Emptying(JointCompressor):
        def process_batch_with_highlights(self, query, docs, threshold, compute_spans, **kwargs):
            return [{"pruned_context": "  ", "reranking_score": self.scores[d]} for d in docs]

    result = rerank_and_compress(Emptying({"a": 0.9}), "q", pool("a"), top_k=1)
    assert result.selected[0].was_emptied
