"""The search tool the agent calls.

Each call is a complete miniature of the integration tier: retrieve 50
candidates, rerank and compress them, hand back the best 5. The difference is
that the agent chooses the query, so the compressor is judged on documents it
was never tuned for, and a bad pruning decision compounds — the agent may
re-query based on what it was shown.

Snippets are capped in tokens because the agent pays for every one it reads, and
an uncapped tool response would let a compressor "win" by returning everything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..pipeline import Candidate, rerank_and_compress

SNIPPET_TOKENIZER = "Qwen/Qwen3-0.6B"


@dataclass
class SearchStats:
    """Per-call accounting, aggregated into the run's compression numbers.

    ``returned_docids`` is what survived reranking and was actually shown to the
    agent, which is what recall is measured on: crediting the retriever's full
    pool would credit the compressor for documents it discarded.

    ``failures`` counts searches that raised. They are reported back to the agent
    rather than aborting the run, so without this count a run in which every
    search failed would summarise as zero compression rather than as broken.
    """

    calls: int = 0
    failures: int = 0
    returned_docids: set[str] = field(default_factory=set)
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def compression_ratio(self) -> float:
        return 1.0 - (self.tokens_after / self.tokens_before) if self.tokens_before else 0.0


class SearchTool:
    """Retrieve, compress, and serialise results for the agent."""

    name = "local_knowledge_base_retrieval"

    def __init__(
        self,
        retriever,
        compressor,
        *,
        top_k: int = 5,
        retrieve_k: int = 50,
        threshold: float = 0.1,
        snippet_max_tokens: int = 512,
        prune: bool = True,
        batch_kwargs: dict | None = None,
    ):
        self.retriever = retriever
        self.compressor = compressor
        self.top_k = top_k
        self.retrieve_k = retrieve_k
        self.threshold = threshold
        self.snippet_max_tokens = snippet_max_tokens
        self.prune = prune
        self.batch_kwargs = batch_kwargs or {}
        self.stats = SearchStats()

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(SNIPPET_TOKENIZER)

    def definition(self) -> dict:
        return {
            "type": "function",
            "name": self.name,
            "description": (
                f"Search the local knowledge base and return the top {self.top_k} most "
                "relevant documents, each with its docid and a text snippet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "Query to search the local knowledge base for relevant information",
                    }
                },
                "required": ["user_query"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def _cap(self, text: str) -> str:
        if not self.snippet_max_tokens:
            return text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.snippet_max_tokens:
            return text
        return self.tokenizer.decode(tokens[: self.snippet_max_tokens], skip_special_tokens=True)

    def __call__(self, query: str) -> str:
        # Counted before the work, so a call that raises still shows up as one
        # the agent made and ``failures`` stays a subset of ``calls``.
        self.stats.calls += 1
        retrieved = self.retriever.search(query, self.retrieve_k)

        result = rerank_and_compress(
            self.compressor,
            query,
            [Candidate(doc_id=r.docid, text=r.text, score=r.score) for r in retrieved],
            top_k=self.top_k,
            threshold=self.threshold,
            prune=self.prune,
            batch_kwargs=self.batch_kwargs,
        )

        results = []
        for item in result.selected:
            snippet = self._cap(item.text)
            self.stats.returned_docids.add(item.doc_id)
            self.stats.tokens_before += len(
                self.tokenizer.encode(item.original_text, add_special_tokens=False)
            )
            self.stats.tokens_after += len(self.tokenizer.encode(snippet, add_special_tokens=False))
            entry = {"docid": item.doc_id, "snippet": snippet}
            if item.score is not None:
                entry["score"] = item.score
            results.append(entry)

        return json.dumps(results, indent=2)
