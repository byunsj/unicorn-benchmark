"""OfficeQA's numeric scorer.

OfficeQA answers are figures pulled out of financial tables, so the headline
metric is not an LLM verdict but agreement with the reference number inside a
tolerance band. The scorer that defines those bands is published by the dataset
authors, and grading with anything else would not be the same benchmark, so it
is fetched from their repository at a pinned commit and checked against a
recorded digest before use.

The bands answer different questions: ``5%`` asks whether the agent found the
right figure at all, ``exact`` whether it read it off precisely.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import urllib.request
from pathlib import Path
from types import ModuleType

# databricks/officeqa @ 7b9a3c154ef9fb40215bb67934afc43e6799de16
SCORER_URL = (
    "https://raw.githubusercontent.com/databricks/officeqa/"
    "7b9a3c154ef9fb40215bb67934afc43e6799de16/reward.py"
)
SCORER_SHA256 = "0d91698c87df6d889339aac36f63ae0966607f169890b0bf8b472b26bfe8138f"

TOLERANCES = {"5%": 0.05, "1%": 0.01, "0.1%": 0.001, "exact": 0.0}

_scorer: ModuleType | None = None


def _cache_path() -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / "officeqa" / "reward.py"


def load_scorer() -> ModuleType:
    """Download, verify and import the official scorer, caching it on disk."""
    global _scorer
    if _scorer is not None:
        return _scorer

    path = _cache_path()
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != SCORER_SHA256:
        with urllib.request.urlopen(SCORER_URL, timeout=30) as response:
            source = response.read()
        digest = hashlib.sha256(source).hexdigest()
        if digest != SCORER_SHA256:
            raise RuntimeError(f"OfficeQA scorer digest mismatch: got {digest}, expected {SCORER_SHA256}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source)

    spec = importlib.util.spec_from_file_location("officeqa_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _scorer = module
    return module


def extract_exact_answer(response: str) -> str:
    """Pull the figure out of the agent's prose, which follows the prompt's format."""
    if not response:
        return ""
    match = re.search(r"Exact Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else response


def numeric_scores(reference: str, predicted: str) -> dict[str, bool]:
    """Score one answer at every tolerance band."""
    score_answer = load_scorer().score_answer
    return {label: bool(score_answer(reference, predicted, tol)) for label, tol in TOLERANCES.items()}
