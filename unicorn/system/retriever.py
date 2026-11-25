"""Dense retrieval over a prebuilt index.

Both system-tier corpora are indexed with the same model, and their shards hold
L2-normalised embeddings, so inner product is cosine similarity and an
exhaustive matmul on the GPU is both exact and fast enough at these corpus sizes
(~100k documents). That removes the need for a FAISS or Tevatron dependency
without changing the ranking: a flat inner-product index is itself a brute-force
scan.

Queries are encoded with the same model that built the index, using last-token
pooling and the retrieval instruction prefix the index was built with. Both must
match or the query and document vectors will not live in the same space.

BrowseComp-Plus uses the index its authors published. OfficeQA's original index
was built with a proprietary embedding model, so this project rebuilt it with
the open model used here; that makes the tier runnable by anyone but puts
OfficeQA in a different retrieval space than the published numbers.
"""

from __future__ import annotations

import glob
import pickle
from dataclasses import dataclass

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

# The instruction the index was built with; changing it silently degrades recall.
TASK_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

MAX_QUERY_TOKENS = 8192


@dataclass(frozen=True)
class IndexSpec:
    index_repo: str
    index_subdir: str
    corpus_repo: str
    corpus_file: str | None = None
    docid_column: str = "docid"


INDEXES = {
    "browsecomp-plus": IndexSpec(
        index_repo="Tevatron/browsecomp-plus-indexes",
        index_subdir="qwen3-embedding-8b",
        corpus_repo="Tevatron/browsecomp-plus-corpus",
    ),
    "officeqa": IndexSpec(
        index_repo="unicorn-rerank/unicorn-officeqa",
        index_subdir="index/qwen3-embedding-8b",
        corpus_repo="unicorn-rerank/unicorn-officeqa",
        corpus_file="corpus.parquet",
    ),
}


@dataclass
class Retrieved:
    docid: str
    text: str
    score: float


class DenseRetriever:
    """Encode a query and return the nearest documents by cosine similarity."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        *,
        dataset: str = "browsecomp-plus",
        index_path: str | None = None,
        device: str = "cuda",
    ):
        if dataset not in INDEXES:
            raise ValueError(f"no index for {dataset!r}. Available: {sorted(INDEXES)}")
        self.spec = INDEXES[dataset]
        self.model_name = model_name
        self.device = device
        self.index_path = index_path
        self.embeddings = None
        self.docids: list[str] = []
        self.corpus: dict[str, str] = {}
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        shards = self._shard_paths()
        vectors, docids = [], []
        for path in shards:
            with open(path, "rb") as f:
                reps, lookup = pickle.load(f)
            vectors.append(torch.from_numpy(reps) if not isinstance(reps, torch.Tensor) else reps)
            docids.extend(lookup)
        self.embeddings = torch.cat(vectors).to(self.device, torch.float32)
        self.docids = [str(d) for d in docids]

        from datasets import load_dataset

        kwargs = {"data_files": self.spec.corpus_file} if self.spec.corpus_file else {}
        corpus = load_dataset(self.spec.corpus_repo, split="train", **kwargs)
        column = self.spec.docid_column
        self.corpus = {str(row[column]): row["text"] for row in corpus}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(self.model_name, dtype=torch.float16).to(self.device).eval()

    def _shard_paths(self) -> list[str]:
        if self.index_path:
            paths = sorted(glob.glob(self.index_path))
            if not paths:
                raise FileNotFoundError(f"no index shards matched {self.index_path!r}")
            return paths

        from huggingface_hub import snapshot_download

        local = snapshot_download(
            repo_id=self.spec.index_repo,
            repo_type="dataset",
            allow_patterns=[f"{self.spec.index_subdir}/*"],
        )
        return sorted(glob.glob(f"{local}/{self.spec.index_subdir}/*.pkl"))

    def encode(self, query: str):
        import torch

        batch = self.tokenizer(
            TASK_PREFIX + query,
            padding=True,
            truncation=True,
            max_length=MAX_QUERY_TOKENS,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state
        # Last-token pooling: with left padding the final position is always real.
        pooled = hidden[:, -1].to(torch.float32)
        return torch.nn.functional.normalize(pooled, dim=-1)

    def search(self, query: str, k: int) -> list[Retrieved]:
        import torch

        if self.embeddings is None:
            raise RuntimeError("call load() before searching")
        scores = self.encode(query) @ self.embeddings.T
        top = torch.topk(scores[0], k=min(k, scores.shape[1]))
        return [
            Retrieved(
                docid=self.docids[idx],
                text=self.corpus.get(self.docids[idx], ""),
                score=float(score),
            )
            for score, idx in zip(top.values.tolist(), top.indices.tolist())
        ]

    def unload(self) -> None:
        import torch

        self.model = None
        self.embeddings = None
        self.corpus = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
