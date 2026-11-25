"""Run the system tier: agent, live retrieval, compression, grading.

    unicorn-system --method comp-h --limit 20

Assumes an OpenAI-compatible server is already hosting the agent model:

    vllm serve openai/gpt-oss-20b --port 8000

The agent and the retriever stay resident for the whole run, since the agent
issues fresh queries continuously. Grading happens afterwards, so the grader
never competes with them for GPU memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..compressors import AVAILABLE_MODELS, create_model
from ..run import ABSTRACTIVE
from .agent import DEFAULT_BASE_URL, DEFAULT_MODEL, Agent
from .data import DATASETS, load_queries
from .grader import GRADER_TEMPLATE, calibration_error, parse_verdict, retrieval_recall
from .retriever import DenseRetriever
from .tool import SearchTool

# OfficeQA answers are figures, so its headline metric is agreement with the
# reference number rather than a judge's verdict.
NUMERIC_DATASETS = {"officeqa"}


def build_parser() -> argparse.ArgumentParser:
    from ..judge import DEFAULT_MODEL as DEFAULT_JUDGE

    p = argparse.ArgumentParser(prog="unicorn-system", description=__doc__.split("\n")[0])
    p.add_argument("--method", required=True, choices=sorted(AVAILABLE_MODELS))
    p.add_argument("--dataset", default="browsecomp-plus", choices=sorted(DATASETS))
    p.add_argument("--threshold", type=float, help="defaults to the method's tuned value")
    p.add_argument("--top-k", type=int, default=5, help="documents returned per search call")
    p.add_argument("--retrieve-k", type=int, default=50, help="candidates retrieved per search call")
    p.add_argument("--snippet-max-tokens", type=int, default=512, help="0 disables the cap")
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=(
            "rows per compressor forward. 0 leaves each method at its own default, "
            "which is what the published runs used; lower it to fit a smaller GPU"
        ),
    )
    p.add_argument("--limit", type=int, help="stop after this many queries")
    p.add_argument("--out", type=Path, default=Path("results/system"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-prune", action="store_true", help="rerank only, leaving documents whole")

    agent = p.add_argument_group("agent")
    agent.add_argument("--agent-model", default=DEFAULT_MODEL)
    agent.add_argument("--agent-url", default=DEFAULT_BASE_URL)
    agent.add_argument("--max-iterations", type=int, default=100)
    agent.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])

    grade = p.add_argument_group("grading")
    grade.add_argument("--no-grade", action="store_true", help="collect trajectories without grading")
    grade.add_argument(
        "--grade-only",
        type=Path,
        help=(
            "grade an existing results file instead of running the agent. The grader "
            "will not fit on the GPU beside the agent server, so stop the server first."
        ),
    )
    grade.add_argument("--judge-model", default=DEFAULT_JUDGE)
    return p


def grade(records: list[dict], judge_model: str) -> None:
    """Judge every trajectory in one batch, writing verdicts back in place."""
    from vllm import SamplingParams

    from ..llm import VLLMEngine

    engine = VLLMEngine(judge_model, max_model_len=8192, enforce_eager=True, memory_cap=0.92)
    try:
        prompts = [
            engine.truncate(
                engine.chat_prompt(
                    [
                        {
                            "role": "user",
                            "content": GRADER_TEMPLATE.format(
                                question=r["query"],
                                response=r["answer"] or "",
                                correct_answer=r["gold_answer"],
                            ),
                        }
                    ]
                ),
                reserve_tokens=512,
            )
            for r in records
        ]
        outputs = engine.generate(prompts, SamplingParams(max_tokens=512, temperature=0.0), desc="Grade")
        for record, output in zip(records, outputs):
            verdict = parse_verdict(output.outputs[0].text)
            record.update(
                extracted_answer=verdict.extracted_answer,
                correct=verdict.correct,
                confidence=verdict.confidence,
                parse_error=verdict.parse_error,
            )
    finally:
        engine.unload()


def score_numeric(records: list[dict]) -> None:
    """Score OfficeQA answers against the reference figure, in place."""
    from .numeric import extract_exact_answer, numeric_scores

    for record in records:
        if record["status"] != "completed" or not record.get("answer"):
            continue
        predicted = extract_exact_answer(record["answer"])
        # The agent does not always follow the answer format; the judge's
        # extraction is a usable fallback when it does not.
        if predicted == record["answer"] and record.get("extracted_answer"):
            predicted = record["extracted_answer"]
        record["numeric"] = numeric_scores(record["gold_answer"], predicted)


def summarize_numeric(records: list[dict]) -> dict:
    scored = [r["numeric"] for r in records if r.get("numeric")]
    if not scored:
        return {}
    from .numeric import TOLERANCES

    return {
        "numeric_accuracy": {
            label: sum(1 for s in scored if s[label]) / len(scored) for label in TOLERANCES
        },
        "numeric_coverage": len(scored),
    }


def summarize(records: list[dict]) -> dict:
    graded = [r for r in records if r.get("correct") is not None]
    recalls = [r["retrieval_recall"] for r in records if r.get("retrieval_recall") is not None]
    confidences = [(r["confidence"], r["correct"]) for r in graded if r.get("confidence") is not None]
    ratios = [r["compression_ratio"] for r in records]

    return {
        "n_queries": len(records),
        "n_graded": len(graded),
        "accuracy": (sum(1 for r in graded if r["correct"]) / len(graded)) if graded else None,
        "retrieval_recall": (sum(recalls) / len(recalls)) if recalls else None,
        "compression_ratio": (sum(ratios) / len(ratios)) if ratios else None,
        "search_calls_mean": sum(r["search_calls"] for r in records) / len(records) if records else 0.0,
        "search_failures": sum(r.get("search_failures", 0) for r in records),
        "calibration_error": calibration_error([c for c, _ in confidences], [ok for _, ok in confidences]),
        "incomplete": sum(1 for r in records if r["status"] != "completed"),
        **summarize_numeric(records),
    }


def _write(out: Path, stem: str, records: list[dict], summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{stem}.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out / f"summary_{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.method in ABSTRACTIVE:
        # The agent reads its search results and decides what to look for next, so
        # a method whose output only its own decoder can read has nothing to hand
        # back. This is a property of the tier, not a gap in the implementation.
        raise SystemExit(
            f"{args.method} compresses into memory embeddings, which the agent cannot read; "
            "the system tier needs methods that return text."
        )
    threshold = (
        args.threshold if args.threshold is not None else AVAILABLE_MODELS[args.method]["default_threshold"]
    )
    stem = f"{args.method}_{args.dataset}_t{threshold:g}".replace("/", "-")
    header = {"method": args.method, "dataset": args.dataset, "threshold": threshold}

    if args.grade_only:
        lines = args.grade_only.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        grade(records, args.judge_model)
        if args.dataset in NUMERIC_DATASETS:
            score_numeric(records)
        _write(args.out, stem, records, {**header, **summarize(records)})
        return 0

    retriever = DenseRetriever(dataset=args.dataset, device=args.device)
    retriever.load()
    compressor = create_model(args.method)
    compressor.load(args.device)

    tool = SearchTool(
        retriever,
        compressor,
        top_k=args.top_k,
        retrieve_k=args.retrieve_k,
        threshold=threshold,
        snippet_max_tokens=args.snippet_max_tokens,
        prune=not args.no_prune,
        # Omitted rather than passed as 0, so each method keeps its own default.
        batch_kwargs={"batch_size": args.batch_size} if args.batch_size > 0 else {},
    )
    agent = Agent(
        tool,
        model=args.agent_model,
        base_url=args.agent_url,
        max_iterations=args.max_iterations,
        reasoning_effort=args.reasoning_effort,
    )

    records: list[dict] = []
    try:
        for query in load_queries(limit=args.limit, dataset=args.dataset):
            before, after = tool.stats.tokens_before, tool.stats.tokens_after
            trajectory = agent.run(query.query_id, query.query)
            spent_before = tool.stats.tokens_before - before
            spent_after = tool.stats.tokens_after - after

            records.append(
                {
                    "query_id": query.query_id,
                    "query": query.query,
                    "gold_answer": query.answer,
                    "answer": trajectory.answer,
                    "status": trajectory.status,
                    "error": trajectory.error,
                    "search_calls": trajectory.search_calls,
                    "search_failures": trajectory.search_failures,
                    "shown_docids": trajectory.shown_docids,
                    "retrieval_recall": retrieval_recall(set(trajectory.shown_docids), query.evidence_docids),
                    "tokens_before": spent_before,
                    "tokens_after": spent_after,
                    "compression_ratio": 1.0 - (spent_after / spent_before) if spent_before else 0.0,
                    "messages": trajectory.messages,
                }
            )
            failed = f" failed={trajectory.search_failures}" if trajectory.search_failures else ""
            print(
                f"[{len(records)}] {query.query_id} {trajectory.status} "
                f"searches={trajectory.search_calls}{failed}",
                flush=True,
            )
    finally:
        compressor.unload()
        retriever.unload()

    failures = sum(r["search_failures"] for r in records)
    if failures:
        # Usually the compressor running out of memory beside the agent server.
        # The agent is told and carries on, so without this the run would look
        # merely unimpressive rather than broken.
        total = sum(r["search_calls"] for r in records)
        print(
            f"warning: {failures} of {total} searches failed; see the records for the error",
            file=sys.stderr,
            flush=True,
        )

    if not args.no_grade and records:
        grade(records, args.judge_model)
        if args.dataset in NUMERIC_DATASETS:
            score_numeric(records)

    _write(args.out, stem, records, {**header, **summarize(records)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
