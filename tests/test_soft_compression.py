"""Comp-S and OSCAR read a bounded prefix of each document and emit no text.

Both properties are silent when they break: a wrong window budget still produces
answers, and a text-based compression ratio still produces a number.
"""

from unicorn.metrics import compression_stats
from unicorn.models.chunking import add_memory_tokens_to_inputs, chunk_doc_to_rows
from unicorn.pipeline import Candidate, rerank_and_generate


class FakeTokenizer:
    """One token per whitespace-separated word, with no special tokens."""

    mem_token_id = -1
    eos_token_id = -2

    def __call__(self, text, **kwargs):
        return {"input_ids": [hash(w) % 1000 for w in text.split()]}

    def encode(self, text, **kwargs):
        return self(text)["input_ids"]

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids)


class FakeOscar:
    """Returns the chunks it was asked to compress, so the mapping can be checked."""

    def __init__(self, scores, chunks_by_doc):
        self.scores = scores
        self.chunks_by_doc = chunks_by_doc

    def rerank_and_generate(self, query, docs, top_k, max_new_tokens=256):
        ranked = sorted(range(len(docs)), key=lambda i: self.scores[i], reverse=True)
        top = ranked[:top_k]
        return {
            "scores": self.scores,
            "ranked_indices": ranked,
            "top_k_indices": top,
            "selected_chunks_by_doc": [self.chunks_by_doc[i] for i in top],
            "n_mem_tokens": 8,
            "generated_answer": "an answer",
        }


class Doc:
    def __init__(self, doc_id, original, kept):
        self.doc_id = doc_id
        self.original_text = original
        self.text = kept


def test_chunking_keeps_only_the_leading_windows():
    doc = " ".join(f"w{i}" for i in range(500))
    rows = chunk_doc_to_rows(
        FakeTokenizer(), "a query", doc, compressor_max_length=128, query_dependent=False, max_chunks_per_doc=3
    )
    assert len(rows) == 3


def test_query_prefix_shrinks_the_window():
    doc = " ".join(f"w{i}" for i in range(500))
    query = " ".join(f"q{i}" for i in range(40))
    plain = chunk_doc_to_rows(
        FakeTokenizer(), query, doc, compressor_max_length=128, query_dependent=False
    )
    prefixed = chunk_doc_to_rows(
        FakeTokenizer(), query, doc, compressor_max_length=128, query_dependent=True
    )
    # The query rides in every window, so fewer document tokens fit in each.
    assert len(prefixed) > len(plain)


def test_short_document_still_produces_one_row():
    rows = chunk_doc_to_rows(
        FakeTokenizer(), "q", "a short document", compressor_max_length=128, query_dependent=False
    )
    assert len(rows) == 1


def test_memory_tokens_scale_with_the_compression_rate():
    rows = [[0] * 32, [0] * 16]
    marked, counts = add_memory_tokens_to_inputs(rows, mem_token_id=-1, compr_rate=16)
    assert counts == [3, 2]
    assert [row.count(-1) for row in marked] == [3, 2]


def test_end_to_end_result_records_what_the_decoder_read():
    pool = [
        Candidate(doc_id="a", text="doc a"),
        Candidate(doc_id="b", text="doc b"),
        Candidate(doc_id="c", text="doc c"),
    ]
    compressor = FakeOscar(scores=[0.1, 0.9, 0.5], chunks_by_doc=[["a1"], ["b1", "b2"], ["c1"]])
    result = rerank_and_generate(compressor, "q", pool, top_k=2)

    assert result.answer == "an answer"
    assert [s.doc_id for s in result.selected] == ["b", "c"]
    # The selected text is the chunks that were compressed, not the whole document.
    assert [s.text for s in result.selected] == ["b1\nb2", "c1"]
    assert result.memory_vectors == 3 * 8


def test_memory_rate_replaces_the_text_token_count():
    selected = [Doc("a", "x " * 320, "x " * 320)]
    stats = compression_stats(selected, "context", FakeTokenizer(), memory_rate=16)

    assert stats["tokens_orig_sum"] == 320
    assert stats["tokens_compressed"] == 20
    assert stats["compression_ratio"] == 0.9375


def test_memory_rate_counts_documents_whole():
    """A document read only in part is still counted whole: the ratio is the rate."""
    truncated = [Doc("a", "x " * 320, "x " * 32)]
    stats = compression_stats(truncated, "context", FakeTokenizer(), memory_rate=16)
    assert stats["tokens_compressed"] == 20
