"""Metrics recorded for every query.

Three families, kept separate because they answer different questions:

- **ranking** — did reranking surface the gold documents at all
- **compression** — how much text survived, in tokens
- **evidence recall** — did the surviving text still contain the gold spans

Evidence recall is the one that distinguishes a compressor that keeps the right
sentences from one that merely keeps few sentences.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

CONTEXT_HEADER = "## Reference Document (ID: {doc_id})"
EMPTY_CONTEXT = "No context."


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip() if text else ""


def build_context(selected: Sequence[Any]) -> str:
    """Join the surviving documents into the generator prompt context.

    Documents pruned to nothing are dropped rather than contributing an empty
    section, which matches how the answer would actually be produced.
    """
    parts = [
        f"{CONTEXT_HEADER.format(doc_id=item.doc_id)}\n{item.text}"
        for item in selected
        if item.text and item.text.strip()
    ]
    return "\n\n".join(parts) if parts else EMPTY_CONTEXT


def evidence_has_weights(evidence: Sequence[dict]) -> bool:
    return any(isinstance(e, dict) and "weight" in e for e in evidence or [])


def evidence_recall(evidence: Sequence[dict], context: str) -> dict[str, Any]:
    """Fraction of gold evidence still present in *context*, by substring containment.

    Datasets whose evidence carries a ``weight`` are scored weighted (a span
    marked by more annotators counts for more); the rest are scored per span.
    Absence of ``weight`` is meaningful, so it must not be defaulted to 1.
    """
    items = list(evidence or [])
    weighted = evidence_has_weights(items)
    metric = "weighted" if weighted else "standard"

    if not items:
        return {
            "evidence_total": 0,
            "evidence_hit_count": 0,
            "evidence_recall": 0.0,
            "evidence_any_retrieved": False,
            "evidence_all_retrieved": False,
            "evidence_hits": [],
            "evidence_metric": metric,
        }

    ctx_norm = _normalize_whitespace(context)
    hits: list[bool] = []
    hit_weight = 0
    total_weight = 0
    for item in items:
        text = (item.get("text") or "").strip() if isinstance(item, dict) else str(item).strip()
        weight = int(item.get("weight", 1)) if (weighted and isinstance(item, dict)) else 1
        if not text:
            hits.append(False)
            continue
        total_weight += weight
        is_hit = bool(ctx_norm) and _normalize_whitespace(text) in ctx_norm
        hits.append(is_hit)
        if is_hit:
            hit_weight += weight

    return {
        "evidence_total": len(items),
        "evidence_hit_count": hit_weight,
        "evidence_recall": (hit_weight / total_weight) if total_weight else 0.0,
        "evidence_any_retrieved": hit_weight > 0,
        "evidence_all_retrieved": total_weight > 0 and hit_weight == total_weight,
        "evidence_hits": hits,
        "evidence_metric": metric,
    }


def evidence_not_applicable(evidence: Sequence[dict]) -> dict[str, Any]:
    """Placeholder for abstractive compressors.

    Comp-S and OSCAR emit memory embeddings, not text, so there is no compressed
    string in which to look for the gold spans. Reporting 0.0 would read as a
    failure rather than an undefined measurement.
    """
    return {
        "evidence_total": len(evidence or []),
        "evidence_hit_count": None,
        "evidence_recall": None,
        "evidence_any_retrieved": None,
        "evidence_all_retrieved": None,
        "evidence_hits": [],
        "evidence_metric": "not_applicable_abstractive",
    }


def ranking_stats(
    ranked_doc_ids: Sequence[str],
    retrieved_doc_ids: Sequence[str],
    gold_doc_ids: set[str],
    top_k: int,
) -> dict[str, Any]:
    """Where the gold documents landed, before and after reranking.

    Ranks are 1-based. The ``retriever_gold_in_top*`` fields describe the pool the
    compressor was handed, which bounds what reranking could recover.
    """
    gold_ranks = [i + 1 for i, doc_id in enumerate(ranked_doc_ids) if doc_id in gold_doc_ids]
    top_ids = list(ranked_doc_ids[:top_k])
    return {
        "retriever_gold_in_top10": any(d in gold_doc_ids for d in retrieved_doc_ids[:10]),
        "retriever_gold_in_top50": any(d in gold_doc_ids for d in retrieved_doc_ids[:50]),
        "rerank_top1_has_gold": bool(top_ids) and top_ids[0] in gold_doc_ids,
        "rerank_gold_in_topk": any(d in gold_doc_ids for d in top_ids),
        "rerank_best_gold_rank": min(gold_ranks) if gold_ranks else None,
        "rerank_all_gold_ranks": gold_ranks,
    }


def compression_stats(
    selected: Sequence[Any],
    context: str,
    tokenizer,
    prefix: str = "",
    memory_rate: int | None = None,
) -> dict[str, Any]:
    """Token counts before and after compression, plus the resulting ratio.

    The ratio is measured on the assembled context rather than per document, so
    it reflects what the generator actually pays for.

    Comp-S and OSCAR pass their *memory_rate* instead: they hand the decoder one
    memory embedding per that many document tokens, and no text at all, so their
    surviving string is not what the model reads and would report no compression.
    """
    orig_by_doc: dict[str, int] = {}
    kept_by_doc: dict[str, int] = {}
    for item in selected:
        orig_by_doc[str(item.doc_id)] = len(tokenizer.encode(item.original_text, add_special_tokens=False))
        kept_by_doc[str(item.doc_id)] = (
            len(tokenizer.encode(item.text, add_special_tokens=False)) if item.text.strip() else 0
        )

    total_orig = sum(orig_by_doc.values())
    if memory_rate:
        # Every selected document is counted whole, because the rate is a property
        # of the compressor rather than of how much of the document it read.
        total_comp = sum(math.ceil(n / memory_rate) for n in orig_by_doc.values() if n > 0)
    elif context != EMPTY_CONTEXT:
        total_comp = len(tokenizer.encode(context, add_special_tokens=False))
    else:
        total_comp = 0
    ratio = 1.0 - (total_comp / total_orig) if total_orig > 0 else 0.0
    return {
        f"{prefix}tokens_orig_sum": total_orig,
        f"{prefix}tokens_compressed": total_comp,
        f"{prefix}compression_ratio": ratio,
        f"{prefix}orig_tokens_by_doc_id": orig_by_doc,
        f"{prefix}kept_tokens_by_doc_id": kept_by_doc,
    }


def aggregate(records: Sequence[dict]) -> dict[str, Any]:
    """Average the per-query records into the numbers a results table reports.

    Fields that are ``None`` (an undefined measurement, not a zero) are skipped
    rather than counted, so abstractive runs do not report a misleading 0.
    """

    def mean(key: str) -> float | None:
        values = [r[key] for r in records if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "n_queries": len(records),
        "judge_score": mean("judge_score"),
        "rerank_gold_in_topk": mean("rerank_gold_in_topk"),
        "rerank_top1_has_gold": mean("rerank_top1_has_gold"),
        "evidence_recall": mean("evidence_recall"),
        "evidence_all_retrieved": mean("evidence_all_retrieved"),
        "compression_ratio": mean("compression_ratio"),
        "tokens_compressed_mean": mean("tokens_compressed"),
        "tokens_orig_mean": mean("tokens_orig_sum"),
    }
