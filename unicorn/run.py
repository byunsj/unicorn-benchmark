"""Run a compressor over a tier and record per-query metrics.

    unicorn --method comp-h --tier integration --dataset hotpotqa
    unicorn --method comp-h --tier integration --dataset hotpotqa --threshold 0.1 0.3 0.5
    unicorn --method comp-h --tier integration --dataset hotpotqa --generate --judge

Threshold controls how aggressively a compressor prunes, and it is the knob worth
sweeping: a single value is one point on the compression/quality curve, while a
list traces the curve. Passing several reuses the loaded model across values.

The stages run one model at a time — compressor, then generator, then judge —
because the judge alone is larger than most GPUs can hold alongside anything
else. Every sweep point is compressed before the generator loads, so a sweep
pays the model-loading cost once rather than once per threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compressors import AVAILABLE_MODELS, create_model
from .data import TIER_DATASETS, load_examples
from .metrics import (
    aggregate,
    build_context,
    compression_stats,
    evidence_not_applicable,
    evidence_recall,
    ranking_stats,
)
from .pipeline import rerank_and_compress, rerank_and_generate

# Compressors that emit memory embeddings rather than text. Their output cannot be
# searched for evidence spans, and their compression is counted in memory vectors.
ABSTRACTIVE = {"comp-s", "oscar"}

# OSCAR reads those embeddings with its own decoder in the same call that
# compresses them, so it answers during compression and skips the generator.
END_TO_END = {"oscar"}

# Comp-S also decodes from its own memory, but as a separate stage: the compressor
# only selects documents, and CompSGenerator compresses and answers them.
COMP_S = "comp-s"

# The oracle: instead of scoring the retrieved pool, it is handed the gold
# documents directly and prunes nothing.
GOLD_ONLY = "gold-only"

# The unit tier gives one document per query, so there is nothing to rank and
# nothing to select: the compressor is judged purely on what it prunes.
TIER_TOP_K = {"unit": 1, "integration": 5}


def build_parser() -> argparse.ArgumentParser:
    from .generator import DEFAULT_MODEL as DEFAULT_GENERATOR
    from .judge import DEFAULT_MODEL as DEFAULT_JUDGE

    p = argparse.ArgumentParser(prog="unicorn", description=__doc__.split("\n")[0])
    p.add_argument("--method", required=True, choices=sorted(AVAILABLE_MODELS), help="compressor to evaluate")
    p.add_argument("--tier", default="integration", choices=sorted(TIER_DATASETS))
    p.add_argument("--dataset", nargs="+", help="datasets within the tier; defaults to all of them")
    p.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        help="keep-probability cutoff; repeat to sweep. Defaults to the method's tuned value.",
    )
    p.add_argument("--top-k", type=int, help="documents kept after reranking; defaults per tier")
    p.add_argument("--rerank-k", type=int, default=50, help="candidates considered; 0 uses the whole pool")
    p.add_argument("--limit", type=int, help="stop after this many queries (for smoke tests)")
    p.add_argument("--batch-size", type=int, default=8, help="documents per compressor forward")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--repo", default=None, help="dataset source; a Hub repo id or a local Parquet directory")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-prune", action="store_true", help="rerank only, leaving documents uncompressed")

    gen = p.add_argument_group("generation and judging")
    gen.add_argument("--generate", action="store_true", help="answer each query from the compressed context")
    gen.add_argument("--judge", action="store_true", help="score answers against gold (implies --generate)")
    gen.add_argument("--generator-model", default=DEFAULT_GENERATOR)
    gen.add_argument("--judge-model", default=DEFAULT_JUDGE)
    gen.add_argument("--max-new-tokens", type=int, default=256)
    return p


def compress_dataset(compressor, *, dataset: str, threshold: float, args, generator_tokenizer) -> list[dict]:
    """Rerank, compress, and measure one dataset at one threshold."""
    examples = load_examples(
        args.tier, dataset, limit=args.limit, **({"repo": args.repo} if args.repo else {})
    )
    abstractive = args.method in ABSTRACTIVE
    end_to_end = args.method in END_TO_END
    # A soft method's compression is the rate at which it turns document tokens
    # into memory embeddings, since it produces no text to measure.
    memory_rate = compressor.compression_rate if abstractive else None
    records: list[dict] = []

    top_k = args.top_k
    for example in examples:
        pool = example.pool(args.rerank_k)
        if args.method == GOLD_ONLY:
            # Every gold document in the candidate window, uncompressed, with no
            # top_k cap: the oracle is defined by perfect retrieval rather than
            # by a document budget, so datasets that annotate more than top_k
            # gold documents keep all of them.
            pool = [c for c in pool if c.doc_id in example.gold_doc_ids]
            top_k = len(pool)

        memory_vectors = None
        if end_to_end:
            result = rerank_and_generate(
                compressor, example.query, pool, top_k=top_k, max_new_tokens=args.max_new_tokens
            )
            memory_vectors = result.memory_vectors
        else:
            result = rerank_and_compress(
                compressor,
                example.query,
                pool,
                top_k=top_k,
                threshold=threshold,
                prune=not args.no_prune,
                batch_kwargs={"batch_size": args.batch_size},
            )
            if args.method == COMP_S:
                memory_vectors = compressor.memory_vectors(
                    example.query, [s.text for s in result.selected]
                )

        context = build_context(result.selected)
        ranked_doc_ids = [pool[i].doc_id for i in result.ranked_indices]
        records.append(
            {
                "query_id": example.query_id,
                "query": example.query,
                "gold_answer": example.answer,
                "gold_doc_ids": sorted(example.gold_doc_ids),
                "selected_doc_ids": [s.doc_id for s in result.selected],
                "strategy": result.strategy,
                "compressed_context": context,
                **({"generated_answer": result.answer} if end_to_end else {}),
                **ranking_stats(
                    ranked_doc_ids,
                    [c.doc_id for c in example.candidates],
                    example.gold_doc_ids,
                    top_k,
                ),
                **(
                    evidence_not_applicable(example.evidence)
                    if abstractive
                    else evidence_recall(example.evidence, context)
                ),
                # What the decoder was really given, which the ratio above does not
                # show: the per-document window cap means a long document is read
                # only up to its first 16 windows.
                **({"memory_vectors": memory_vectors} if abstractive else {}),
                **compression_stats(result.selected, context, compressor.tokenizer, memory_rate=memory_rate),
                # Also measured in the generator's vocabulary, since that is the
                # token count the answering model is actually billed for.
                **compression_stats(
                    result.selected, context, generator_tokenizer, prefix="generator_", memory_rate=memory_rate
                ),
            }
        )

    return records


def run_compression(
    args, thresholds: list[float], datasets: list[str]
) -> list[tuple[str, float, list[dict]]]:
    from transformers import AutoTokenizer

    generator_tokenizer = AutoTokenizer.from_pretrained(args.generator_model, trust_remote_code=True)
    compressor = create_model(args.method)
    compressor.load(args.device)
    runs = []
    try:
        for dataset in datasets:
            for threshold in thresholds:
                records = compress_dataset(
                    compressor,
                    dataset=dataset,
                    threshold=threshold,
                    args=args,
                    generator_tokenizer=generator_tokenizer,
                )
                runs.append((dataset, threshold, records))
                print(f"compressed {dataset} t={threshold:g} ({len(records)} queries)", flush=True)
    finally:
        compressor.unload()
    return runs


def run_generation(args, records: list[dict]) -> None:
    """Answer every query in place, across all sweep points in one batch."""
    from .generator import CompSGenerator, Generator

    if args.method == COMP_S:
        # The shared generator would read the selected documents as plain text and
        # report Comp-S as if it had compressed nothing.
        settings = AVAILABLE_MODELS[COMP_S]["init_kwargs"]
        generator = CompSGenerator(
            settings["model_name"],
            max_new_tokens=args.max_new_tokens,
            compressor_max_length=settings["compressor_max_length"],
            max_chunks_per_doc=settings["max_chunks_per_doc"],
            batch_size=args.batch_size,
        )
    else:
        generator = Generator(args.generator_model, max_new_tokens=args.max_new_tokens)
    try:
        outputs = generator.generate(
            [r["query"] for r in records],
            [r["compressed_context"] for r in records],
        )
        for record, output in zip(records, outputs):
            record.update(output)
    finally:
        generator.unload()


def run_judging(args, records: list[dict]) -> str:
    """Score every generated answer in place; returns the metric key used."""
    from .judge import Judge

    judge = Judge(args.judge_model)
    try:
        scores = judge.score(
            [r["query"] for r in records],
            [r["gold_answer"] or "" for r in records],
            [r.get("generated_answer") or "" for r in records],
        )
        for record, score in zip(records, scores):
            record["judge_score"] = score
        return judge.metric_key()
    finally:
        judge.unload()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.judge:
        args.generate = True
    if args.top_k is None:
        args.top_k = TIER_TOP_K[args.tier]

    thresholds = args.threshold or [AVAILABLE_MODELS[args.method]["default_threshold"]]
    datasets = args.dataset or TIER_DATASETS[args.tier]
    unknown = set(datasets) - set(TIER_DATASETS[args.tier])
    if unknown:
        raise SystemExit(f"not in the {args.tier} tier: {sorted(unknown)}")

    runs = run_compression(args, thresholds, datasets)

    all_records = [record for _, _, records in runs for record in records]
    judge_key = None
    if args.generate and all_records and args.method not in END_TO_END:
        run_generation(args, all_records)
    if args.judge and all_records:
        judge_key = run_judging(args, all_records)

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset, threshold, records in runs:
        stem = f"{args.method}_{args.tier}_{dataset}_t{threshold:g}".replace("/", "-")
        with (args.out / f"{stem}.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary = {
            "method": args.method,
            "tier": args.tier,
            "dataset": dataset,
            "threshold": threshold,
            **({"judge_metric": judge_key} if judge_key else {}),
            **aggregate(records),
        }
        summaries.append(summary)
        line = f"{dataset:<26} t={threshold:<5g} "
        # With a single candidate there is nothing to rank, and the oracle is
        # given the gold documents, so in both cases the ranking column would
        # read 100% by construction and tell the reader nothing.
        if args.tier != "unit" and args.method != GOLD_ONLY:
            line += f"gold@{args.top_k}={_pct(summary['rerank_gold_in_topk'])} "
        line += f"evidence={_pct(summary['evidence_recall'])} compression={_pct(summary['compression_ratio'])}"
        if summary.get("judge_score") is not None:
            line += f" accuracy={_pct(summary['judge_score'])}"
        print(line, flush=True)

    (args.out / f"summary_{args.method}_{args.tier}.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    return 0


def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{100 * value:5.1f}"


if __name__ == "__main__":
    sys.exit(main())
