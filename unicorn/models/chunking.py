"""Building ``<MEM>``-marked compressor rows for Comp-S.

Shared by the compressor and the generator so both produce identical ``<MEM>``
layouts; a mismatch here silently misaligns the memory embeddings.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase


def add_memory_tokens_to_inputs(
    input_ids: list[list[int]],
    mem_token_id: int,
    compr_rate: int,
    append_token_id: int | None = None,
) -> tuple[list[list[int]], list[int]]:
    """Append ``len(seq)//compr_rate + 1`` ``<MEM>`` tokens to each sequence.

    ``append_token_id`` adds one trailing token, which protects backbones that drop
    the representation of their last input position (e.g. BidirLM) from losing a
    ``<MEM>`` hidden state.
    """
    n_mems = [len(elt) // compr_rate + 1 for elt in input_ids]
    out = [elt + [mem_token_id] * n_mem for elt, n_mem in zip(input_ids, n_mems)]
    if append_token_id is not None:
        out = [elt + [append_token_id] for elt in out]
    return out, n_mems


def chunk_list(data: list[int], chunk_length: int, chunk_overlap: int = 0) -> list[list[int]]:
    if chunk_overlap >= chunk_length:
        raise ValueError("chunk_overlap must be smaller than chunk_length")
    step = chunk_length - chunk_overlap
    return [
        data[start : start + chunk_length]
        for start in range(0, len(data), step)
        if data[start : start + chunk_length]
    ]


def chunk_doc_to_rows(
    compressor_tokenizer: PreTrainedTokenizerBase,
    query: str,
    doc: str,
    compressor_max_length: int,
    query_dependent: bool,
    max_chunks_per_doc: int = 0,
    special_token_margin: int = 8,
) -> list[list[int]]:
    """Split one document into query-prefixed compressor rows, one per window.

    Rather than truncating at ``compressor_max_length``, the document is cut into
    non-overlapping windows sized to leave room for the query prefix, and each
    window is re-tokenized the same way a single window would be. Always returns at
    least one row.
    """
    tok = compressor_tokenizer
    doc_ids = tok(doc, add_special_tokens=False, truncation=False)["input_ids"]
    if query_dependent:
        prefix = f"Query: {query}\n Document: "
        prefix_len = len(tok(prefix, add_special_tokens=False)["input_ids"])
    else:
        prefix = ""
        prefix_len = 0

    budget = max(1, compressor_max_length - prefix_len - special_token_margin)
    windows = chunk_list(doc_ids, budget) or [[]]
    if max_chunks_per_doc:
        windows = windows[:max_chunks_per_doc]

    rows: list[list[int]] = []
    for window in windows:
        chunk_text = tok.decode(window) if window else ""
        chunk_str = f"{prefix}{chunk_text}" if query_dependent else chunk_text
        rows.append(
            tok(
                chunk_str,
                add_special_tokens=True,
                truncation=True,
                max_length=compressor_max_length,
                return_tensors=None,
            )["input_ids"]
        )
    return rows


__all__ = ["add_memory_tokens_to_inputs", "chunk_doc_to_rows", "chunk_list"]
