"""Compressors evaluated by UniCoRn.

Every entry exposes the same interface (:class:`CompressionModel`): given a query
and a document, return a relevance score and a pruned version of the document.
That is what lets the three tiers swap methods without changing the pipeline.

Each entry names its class as ``module:attribute`` rather than importing it, so
listing the available methods costs nothing and the heavy machine-learning stack
is only imported for the one method actually being run.
"""

from __future__ import annotations

import importlib

from .base import DEFAULT_BGE_RERANKER_MODEL, CompressionModel, build_spans_from_pruned

# Comp-H was trained with these segment affixes; they must match at inference.
_COMP_H_AFFIXES = {
    "query_prefix": "Query: ",
    "doc_prefix": "<|endoftext|> Document: ",
    "doc_suffix": "<|endoftext|>",
}

AVAILABLE_MODELS = {
    "comp-h": {
        "class": "comp_h:CompH",
        "display_name": "Comp-H (ours)",
        "default_threshold": 0.3,
        "init_kwargs": {
            "model_name": "unicorn-rerank/unicorn-comp-h",
            "threshold": 0.3,
            "lazy": True,
            **_COMP_H_AFFIXES,
        },
    },
    "comp-s": {
        "class": "comp_s:CompS",
        "display_name": "Comp-S (ours)",
        # Comp-S has no text-pruning threshold; kept at 0 so run names stay uniform.
        "default_threshold": 0.0,
        "init_kwargs": {
            "model_name": "unicorn-rerank/unicorn-comp-s",
            "lazy": True,
            "compressor_max_length": 128,
            "max_chunks_per_doc": 16,
            "rerank_batch_size": 8,
            "external_reranker_name": DEFAULT_BGE_RERANKER_MODEL,
            "external_reranker_batch_size": 16,
            "external_reranker_max_length": 8192,
        },
    },
    "provence": {
        "class": "provence:ProvenceReranker",
        "display_name": "Provence (DeBERTa-v3)",
        "default_threshold": 0.1,
        "init_kwargs": {"model_name": "naver/provence-reranker-debertav3-v1", "threshold": 0.1, "lazy": True},
    },
    "xprovence": {
        "class": "provence:ProvenceReranker",
        "display_name": "XProvence (BGE-M3-v2)",
        "default_threshold": 0.3,
        "init_kwargs": {"model_name": "naver/xprovence-reranker-bgem3-v2", "threshold": 0.3, "lazy": True},
    },
    "open-provence": {
        "class": "provence:ProvenceReranker",
        "display_name": "OpenProvence (Large, 310M)",
        "default_threshold": 0.1,
        "init_kwargs": {"model_name": "hotchpotch/open-provence-reranker-large-v1", "threshold": 0.1, "lazy": True},
    },
    "oscar": {
        "class": "oscar:OscarPipeline",
        "display_name": "OSCAR (Qwen2-7B)",
        # OSCAR has no text-pruning threshold either; see comp-s above.
        "default_threshold": 0.0,
        # Same window and per-document budget as Comp-S, so the two soft methods
        # compress the same span of the same documents.
        "init_kwargs": {"chunk_max_tokens": 128, "max_chunks_per_doc": 16},
    },
    "zilliz": {
        "class": "zilliz:ZillizSemanticHighlightModel",
        "display_name": "Zilliz Semantic Highlight (Bilingual)",
        "default_threshold": 0.3,
        "init_kwargs": {},
    },
    "zilliz+bge": {
        "class": "zilliz_bge_rerank:ZillizBGERerankModel",
        "display_name": "Zilliz Semantic Highlight + BGE rerank",
        "default_threshold": 0.3,
        "init_kwargs": {"threshold": 0.3},
    },
    "llmlingua-2": {
        "class": "llmlingua2:LLMLingua2Model",
        "display_name": "LLMLingua-2 (XLM-RoBERTa-large)",
        "default_threshold": 0.5,
        "init_kwargs": {},
    },
    # LLMLingua-2 is query-agnostic and has no reranking head, so pairing it with
    # the external BGE selector is what makes a rerank+prune row possible.
    "llmlingua-2+bge": {
        "class": "llmlingua2:LLMLingua2Model",
        "display_name": "LLMLingua-2 + BGE rerank",
        "default_threshold": 0.5,
        "init_kwargs": {"external_reranker_name": DEFAULT_BGE_RERANKER_MODEL},
    },
    "longllmlingua": {
        "class": "longllmlingua:LongLLMLinguaModel",
        "display_name": "LongLLMLingua (LLaMA-2-7B)",
        "default_threshold": 0.55,
        "init_kwargs": {"model_name": "NousResearch/Llama-2-7b-hf"},
    },
    "zerank-DSLR": {
        "class": "zerank_dslr:ZerankDSLR",
        "display_name": "zerank-DSLR (sentence-level, calibrated threshold)",
        "default_threshold": 0.4,
        "init_kwargs": {"model_name": "zeroentropy/zerank-2", "threshold": 0.4},
    },
    # Reference points: rerank without pruning, and no model at all.
    "bge-reranker": {
        "class": "bge_reranker:BGERerankerModel",
        "display_name": "BGE reranker v2-m3 (rerank only)",
        "default_threshold": 0.0,
        "init_kwargs": {},
    },
    "gold-only": {
        "class": "gold_only:GoldOnlyOracle",
        "display_name": "Gold oracle",
        "default_threshold": 0.0,
        "init_kwargs": {},
    },
}


def create_model(name: str) -> CompressionModel:
    """Instantiate a registered compressor. The result is not loaded; call ``.load()``."""
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown compressor: {name}. Available: {sorted(AVAILABLE_MODELS)}")
    entry = AVAILABLE_MODELS[name]
    module_name, _, attribute = entry["class"].partition(":")
    module = importlib.import_module(f".{module_name}", __package__)
    kwargs = dict(entry.get("init_kwargs", {}))
    kwargs.setdefault("display_name", entry.get("display_name", name))
    return getattr(module, attribute)(**kwargs)


__all__ = [
    "AVAILABLE_MODELS",
    "CompressionModel",
    "build_spans_from_pruned",
    "create_model",
]
