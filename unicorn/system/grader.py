"""Grading agent trajectories.

The agent answers in prose, so the grader first extracts a final answer and then
judges it against the reference. Unlike the other tiers this is a text verdict
rather than a logit score, because the grader also has to pull out the answer and
the agent's stated confidence.

Three things are measured, and they can disagree in informative ways: accuracy is
whether the answer was right, retrieval recall is whether the agent was ever shown
the documents needed to justify it, and calibration is whether its confidence
matched its correctness. A compressor that prunes too hard tends to show high
recall with low accuracy — the agent was handed the right document but not the
right sentence.

Note that recall here is document-level. BrowseComp-Plus labels whole documents as
evidence, not spans within them, so the span-level evidence recall reported for
the other two tiers cannot be computed for this one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()


@dataclass
class Verdict:
    extracted_answer: str | None = None
    correct: bool | None = None
    confidence: float | None = None
    parse_error: bool = False


def _search_field(text: str, field: str, value_pattern: str) -> re.Match | None:
    # The grader is inconsistent about markdown bolding, so accept both forms.
    for pattern in (rf"\*\*{field}:?\*\*:?\s*{value_pattern}", rf"{field}:\s*{value_pattern}"):
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match
    return None


def parse_verdict(response: str) -> Verdict:
    if not response:
        return Verdict(parse_error=True)

    verdict = Verdict()
    if match := _search_field(response, "extracted_final_answer", r"(.*?)(?=\n|$)"):
        verdict.extracted_answer = match.group(1).strip()
    if match := _search_field(response, "correct", r"(yes|no)"):
        verdict.correct = match.group(1).lower() == "yes"
    if match := _search_field(response, "confidence", r"(\d+(?:\.\d+)?)\s*%?"):
        verdict.confidence = min(float(match.group(1)), 100.0)

    # Without a verdict there is nothing to score, so flag it rather than
    # silently counting the answer as wrong.
    verdict.parse_error = verdict.correct is None
    return verdict


def retrieval_recall(shown: set[str], evidence: set[str]) -> float | None:
    """Fraction of the evidence-labelled documents the agent was shown.

    *shown* is what the search tool returned after compression, not the retriever's
    full pool, so a compressor that drops a relevant document is penalised here.
    """
    if not evidence:
        return None
    return len(shown & evidence) / len(evidence)


def calibration_error(confidences: list[float], correct: list[bool], bins: int = 10) -> float | None:
    """Expected calibration error between stated confidence and correctness.

    Needs a reasonable sample to mean anything; below one example per bin the
    number is noise, so return None instead.
    """
    if len(confidences) < bins:
        return None

    total = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        members = [
            (c / 100.0, ok)
            for c, ok in zip(confidences, correct)
            if low <= c / 100.0 < high or (b == bins - 1 and c / 100.0 == 1.0)
        ]
        if not members:
            continue
        mean_confidence = sum(c for c, _ in members) / len(members)
        accuracy = sum(1 for _, ok in members if ok) / len(members)
        total += (len(members) / len(confidences)) * abs(mean_confidence - accuracy)
    return total
