"""Loading UniCoRn evaluation data from the Hub.

Each tier is a config and each dataset a split, so only the requested dataset is
downloaded:

    load_dataset("unicorn-rerank/unicorn-benchmark", "integration", split="hotpotqa")
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field

from .pipeline import Candidate

DEFAULT_REPO = "unicorn-rerank/unicorn-benchmark"

UNIT_DATASETS = [
    "multi_hop",
    "hard_to_compress",
    "scattered_evidence",
    "information_density",
    "adversarial_distractor",
    "contextualized",
    "union",
    "broad_open_ended",
    "ugly_formatting",
    "cross_lingual",
]

INTEGRATION_DATASETS = [
    "academic_papers",
    "conditionalqa",
    "finqa",
    "hotpotqa",
    "enterprise_rag",
    "jsquad",
    "korquad",
    "klue_mrc",
    "legalquad",
    "vidore_computer_science",
    "vidore_energy",
    "vidore_finance_en",
    "vidore_finance_fr",
    "vidore_hr",
    "vidore_industrial",
    "vidore_pharmaceuticals",
    "vidore_physics",
]

TIER_DATASETS = {
    "unit": UNIT_DATASETS,
    "integration": INTEGRATION_DATASETS,
}


@dataclass
class Example:
    query_id: str
    query: str
    answer: str
    candidates: list[Candidate]
    qrels: dict[str, int]
    evidence: list[dict]
    query_types: list[str] = field(default_factory=list)
    source: str | None = None
    meta: dict | None = None

    @property
    def gold_doc_ids(self) -> set[str]:
        return {doc_id for doc_id, rel in self.qrels.items() if rel > 0}

    @property
    def evidence_texts(self) -> list[str]:
        return [e["text"].strip() for e in self.evidence if e.get("text")]

    def pool(self, rerank_k: int) -> list[Candidate]:
        return self.candidates if rerank_k <= 0 else self.candidates[:rerank_k]


def _clean_evidence(items: list[dict] | None) -> list[dict]:
    # Parquet stores absent keys as null; drop them so "weight" stays absent.
    return [{k: v for k, v in item.items() if v is not None} for item in (items or [])]


def to_example(row: dict) -> Example:
    scores = row.get("doc_scores") or [None] * len(row["doc_ids"])
    meta = row.get("meta")
    return Example(
        query_id=row["query_id"],
        query=row["query"],
        answer=row.get("answer") or "",
        candidates=[
            Candidate(doc_id=doc_id, text=text, score=score)
            for doc_id, text, score in zip(row["doc_ids"], row["docs"], scores)
        ],
        qrels={q["doc_id"]: q["relevance"] for q in row["qrels"]},
        evidence=_clean_evidence(row.get("evidence")),
        query_types=row.get("query_types") or [],
        source=row.get("source"),
        meta=json.loads(meta) if isinstance(meta, str) else meta,
    )


def load_examples(
    tier: str,
    dataset: str,
    *,
    repo: str = DEFAULT_REPO,
    limit: int | None = None,
) -> Iterator[Example]:
    """Stream one tier/dataset. *repo* may also be a local directory of Parquet files."""
    from datasets import load_dataset

    if tier not in TIER_DATASETS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(TIER_DATASETS)}")

    data = load_dataset(repo, tier, split=dataset)
    for i, row in enumerate(data):
        if limit is not None and i >= limit:
            return
        yield to_example(row)
