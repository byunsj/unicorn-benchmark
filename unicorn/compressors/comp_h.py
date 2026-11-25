"""Comp-H: extractive compressor that reranks and prunes in one pass.

Documents are sentence-split with SaT (``sat-3l-sm``) and greedily packed into
chunks of up to ``max_chunk_tokens`` tokens (default 2048, the training
``max_seq_len``). Chunks across the input batch are flattened into a single
padded forward pass — sub-batched by ``batch_size`` — and per-sentence keep
probabilities are stitched back into each document's original order. A
document's ranking score is the maximum over its chunks, as in OpenProvence.

``model_name`` may be a Hub repo id or a local directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification, PretrainedConfig, PreTrainedModel

# Registers the `bidirlm` backbone with the Auto* classes, which must happen
# before this checkpoint can be rebuilt from its config.
from ..models import bidirlm  # noqa: F401
from ..models.attention import select_attn_implementation
from .base import CompressionModel

logger = logging.getLogger(__name__)


# Vendored CompressionEncoder so this module only depends on torch + transformers.
class _CompressionConfig(PretrainedConfig):
    model_type = "compression_encoder"

    def __init__(
        self,
        base_config_dict: dict[str, Any] | None = None,
        compression_dropout: float = 0.1,
        query_prefix: str = "",
        doc_prefix: str = "",
        doc_suffix: str = "",
        token_head_offset: int = 0,
        cls_token_index: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.base_config_dict = base_config_dict or {}
        self.compression_dropout = compression_dropout
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.doc_suffix = doc_suffix
        # Number of leading hidden-state positions to drop before the per-token
        # compression head. Some backbones (e.g. BidirLM) prepend a token (BOS) and
        # drop the last hidden state, so ``last_hidden_state[:, j]`` is the
        # representation of input token ``j-1``. Read from the saved checkpoint config.
        self.token_head_offset = token_head_offset
        # Position pooled for the ranking logit. Training's CompressionEncoder pools
        # ``last_hidden_state[:, cls_token_index]`` through the backbone's score head
        # when this is set (e.g. Cobble trained with cls_token_index=0). Must be
        # honoured at eval time or the ranking score is pooled from the wrong token.
        self.cls_token_index = cls_token_index


class _CompressionEncoder(PreTrainedModel):
    config_class = _CompressionConfig

    _SCORE_HEAD_NAMES = ("score", "classifier")

    def __init__(self, config: _CompressionConfig):
        super().__init__(config)
        base_cfg_kwargs = {**config.base_config_dict, "attn_implementation": select_attn_implementation()}
        base_config = AutoConfig.for_model(**base_cfg_kwargs)
        self.ranking_model = AutoModelForSequenceClassification.from_config(base_config)
        self.compression_head = nn.Sequential(
            nn.Dropout(config.compression_dropout),
            nn.Linear(base_config.hidden_size, 2),
        )

    def _score_head(self) -> nn.Module:
        for name in self._SCORE_HEAD_NAMES:
            head = getattr(self.ranking_model, name, None)
            if head is not None:
                return head
        raise AttributeError(
            f"Cannot find classification head on {type(self.ranking_model).__name__}; "
            f"tried {self._SCORE_HEAD_NAMES}"
        )

    def resize_token_embeddings(self, new_num_tokens: int | None = None, **kwargs: Any):
        return self.ranking_model.resize_token_embeddings(new_num_tokens, **kwargs)

    def get_input_embeddings(self):
        return self.ranking_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.ranking_model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        cls_token_index = getattr(self.config, "cls_token_index", None)
        if cls_token_index is not None:
            # cls-pooling path: mirror training's CompressionEncoder, which pools
            # ``last_hidden_state[:, cls_token_index]`` through the backbone's score head
            # (e.g. Cobble trained with cls_token_index=0). Takes priority over
            # ``compute_pooled_logits`` to match the training forward's branch order.
            base_model = getattr(self.ranking_model, self.ranking_model.base_model_prefix)
            base_out = base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            hidden_states = base_out.last_hidden_state
            pooled = hidden_states[:, cls_token_index, :]
            ranking_logits = self._score_head()(pooled)
        elif hasattr(self.ranking_model, "compute_pooled_logits"):
            # BidirLM path: mirror training's CompressionEncoder -- run the backbone once,
            # read ``last_hidden_state``, and pool via the model's own head. Avoids
            # ``output_hidden_states`` (a fragile, memory-heavy path on this backbone).
            base_model = getattr(self.ranking_model, self.ranking_model.base_model_prefix)
            base_out = base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            hidden_states = base_out.last_hidden_state
            ranking_logits = self.ranking_model.compute_pooled_logits(hidden_states, attention_mask)
        else:
            outputs = self.ranking_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )
            ranking_logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]

        if ranking_logits.dim() > 1:
            ranking_logits = ranking_logits[:, 0]

        # Realign per-token hidden states with input token positions for backbones
        # that prepend a leading token (see ``_CompressionConfig.token_head_offset``).
        # After slicing, ``token_hidden[:, i]`` is the representation of input token ``i``;
        # right-pad the ``offset`` dropped trailing tokens with a low keep-prob so the
        # returned ``compression_logits`` stays aligned 1:1 with ``input_ids`` / offsets.
        offset = getattr(self.config, "token_head_offset", 0) or 0
        token_hidden = hidden_states[:, offset:, :] if offset else hidden_states
        compression_logits = self.compression_head(token_hidden)
        if offset:
            b, _, c = compression_logits.shape
            pad = compression_logits.new_zeros(b, offset, c)
            pad[..., 1] = -1e4  # keep-class logit -> keep prob ~0 for dropped trailing tokens
            compression_logits = torch.cat([compression_logits, pad], dim=1)
        return {
            "ranking_logits": ranking_logits,
            "compression_logits": compression_logits,
        }


def _resolve_model_path(model_name_or_path: str) -> Path:
    """Return a local directory for a checkpoint, downloading from the Hub if needed."""
    path = Path(model_name_or_path)
    if path.is_dir():
        return path

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model_name_or_path))


class CompH(CompressionModel):
    """Sentence-level hard compression with chunked + batched inference.

    Long documents are split at sentence boundaries into
    ``max_chunk_tokens``-sized pieces; chunks across the input batch are
    forwarded together (sub-batched by ``batch_size``). Token-level keep
    probabilities are aggregated to sentences via SaT ``sat-3l-sm`` so the
    output is comparable to Provence / Zilliz.
    """

    SAT_MODEL_NAME = "sat-3l-sm"
    MAX_CHUNK_TOKENS = 2048  # training max_seq_len
    SPECIAL_TOKEN_BUDGET = 8  # BOS/SEP/EOS on both segments

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.5,
        display_name: str | None = None,
        lazy: bool = True,
        max_chunk_tokens: int | None = None,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
        doc_suffix: str | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.threshold = threshold
        self.display_name = display_name or model_name
        self.max_chunk_tokens = max_chunk_tokens or self.MAX_CHUNK_TOKENS
        self.model: _CompressionEncoder | None = None
        self._tokenizer = None
        self._sentencizer = None
        self.device: str | None = None
        self._query_prefix_override = query_prefix
        self._doc_prefix_override = doc_prefix
        self._doc_suffix_override = doc_suffix
        self.query_prefix = ""
        self.doc_prefix = ""
        self.doc_suffix = ""

        if not lazy:
            self.load()

    def load(self, device: str = "auto") -> None:
        from transformers import AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        local_path = _resolve_model_path(self.model_name)
        local_path_str = str(local_path)
        self._tokenizer = AutoTokenizer.from_pretrained(local_path_str, padding_side="right")

        cfg_file = Path(local_path_str) / "config.json"
        legacy_missing_affix_meta = True
        if cfg_file.is_file():
            with open(cfg_file, encoding="utf-8") as f:
                cfg_json = json.load(f)
            legacy_missing_affix_meta = "query_prefix" not in cfg_json

        # Sync vocab_size with the tokenizer: training may have resized
        # embeddings (e.g. added a pad token) while base_config_dict still
        # holds the original size.
        config = _CompressionConfig.from_pretrained(local_path_str)
        actual_vocab = len(self._tokenizer)
        if config.base_config_dict.get("vocab_size") != actual_vocab:
            config.base_config_dict["vocab_size"] = actual_vocab

        self.query_prefix = (
            self._query_prefix_override
            if self._query_prefix_override is not None
            else getattr(config, "query_prefix", "")
        )
        self.doc_prefix = (
            self._doc_prefix_override if self._doc_prefix_override is not None else getattr(config, "doc_prefix", "")
        )
        self.doc_suffix = (
            self._doc_suffix_override if self._doc_suffix_override is not None else getattr(config, "doc_suffix", "")
        )

        logger.info(
            "CoCompress (chunked) affixes: query_prefix=%r doc_prefix=%r doc_suffix=%r",
            self.query_prefix,
            self.doc_prefix,
            self.doc_suffix,
        )
        if self.query_prefix == "" and self.doc_prefix == "" and self.doc_suffix == "" and legacy_missing_affix_meta:
            logger.warning(
                "CompH (chunked): checkpoint config has no saved query/doc affixes and none were "
                "passed to the constructor. If training used query_prefix/doc_prefix/doc_suffix, "
                "pass them explicitly or re-save the checkpoint with a newer trainer."
            )

        # bfloat16 is the training/GPU dtype; on CPU it mixes badly with the
        # float32 tensors created by resize_token_embeddings (and CPU bf16
        # matmul support is patchy), so load in float32 when not on CUDA.
        model_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = _CompressionEncoder.from_pretrained(
            local_path_str,
            config=config,
            dtype=model_dtype,
            # Materialize everything on CPU instead of via meta-device init;
            # some checkpoints otherwise leave tensors on the meta device, which
            # then crashes the .to() below ("Cannot copy out of meta tensor").
            low_cpu_mem_usage=False,
        )
        # base_config_dict may carry torch_dtype=bfloat16, which makes the
        # from_config-built ranking_model bf16 even when we ask for float32.
        # Cast the whole module (ranking_model + compression_head) uniformly.
        self.model = self.model.to(device=device, dtype=model_dtype).eval()

        # Padded batched inference needs a pad token; some checkpoints don't
        # ship one, so add it and resize embeddings if necessary.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.ranking_model.config.pad_token_id = self.tokenizer.pad_token_id

        try:
            from wtpsplit import SaT
        except ImportError as e:
            raise RuntimeError(
                "wtpsplit is required for sentence splitting. " "Install with: pip install wtpsplit"
            ) from e

        self._sentencizer = SaT(self.SAT_MODEL_NAME)
        if device.startswith("cuda"):
            self._sentencizer.half().to(device)

    def unload(self) -> None:
        del self.model
        del self._tokenizer
        del self._sentencizer
        self.model = None
        self._tokenizer = None
        self._sentencizer = None
        self.device = None
        self.query_prefix = ""
        self.doc_prefix = ""
        self.doc_suffix = ""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_info(self) -> dict:
        return {
            "name": self.model_name,
            "display_name": self.display_name,
            "loaded": self.is_loaded(),
            "device": str(self.device) if self.device else None,
            "max_chunk_tokens": self.max_chunk_tokens,
        }

    @property
    def tokenizer(self):
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        return self._tokenizer

    def _find_doc_start_token(self, encoded: Any, idx: int) -> int:
        """Return the token index where the document (second segment) begins.

        Uses the fast tokenizer's ``sequence_ids`` (segment 1 = document) which is
        robust for tokenizers without a separator token and where ``eos == pad``
        (e.g. Qwen3 / BidirLM). Falls back to a sep/eos scan if unavailable.
        """
        try:
            seq_ids = encoded.sequence_ids(idx)
        except (AttributeError, TypeError):
            seq_ids = None
        if seq_ids is not None:
            for pos, sid in enumerate(seq_ids):
                if sid == 1:
                    return pos

        input_ids = encoded["input_ids"][idx]
        sep_token_id = getattr(self._tokenizer, "sep_token_id", None)
        eos_token_id = self._tokenizer.eos_token_id or 2
        target = sep_token_id if sep_token_id is not None else eos_token_id
        positions = (input_ids == target).nonzero(as_tuple=True)[0]
        if len(positions) >= 1:
            return int(positions[0].item()) + 1
        return 0

    @staticmethod
    def _sents_to_ranges(text: str, sents: list[str]) -> list[tuple[int, int]]:
        """Convert a list of sentence strings back into ``(start, end)`` char ranges."""
        ranges: list[tuple[int, int]] = []
        pos = 0
        for sent in sents:
            if not sent:
                continue
            # SaT preserves characters, but use find() as a guard in case
            # of any whitespace normalisation at boundaries.
            start = text.find(sent, pos)
            if start == -1:
                start = pos
            end = start + len(sent)
            ranges.append((start, end))
            pos = end
        return ranges

    def _sentence_ranges(self, text: str) -> list[tuple[int, int]]:
        """Split ``text`` into sentences and return their ``(start, end)`` char ranges."""
        sents = self._sentencizer.split(text)
        return self._sents_to_ranges(text, sents)

    def _sentence_ranges_batch(self, texts: list[str]) -> list[list[tuple[int, int]]]:
        """Batch sentence splitting via SaT: one call for all texts."""
        all_sents = list(self._sentencizer.split(texts))
        return [self._sents_to_ranges(text, sents) for text, sents in zip(texts, all_sents)]

    def _plan_chunks(
        self,
        query: str,
        doc: str,
        sentence_ranges: list[tuple[int, int]],
    ) -> list[list[int]]:
        """Greedy sentence-aware packing: never split a sentence across chunks.

        Returns a list of chunks; each chunk is a list of sentence indices
        into ``sentence_ranges``. Token-length budgeting accounts for the
        query and special tokens shared with every chunk.
        """
        query_for_budget = self.query_prefix + query
        query_tokens = len(self._tokenizer.encode(query_for_budget, add_special_tokens=False))
        # Second segment in the real forward is doc_prefix + chunk + doc_suffix; count fixed affix tokens once.
        doc_affix_overhead = len(self._tokenizer.encode(self.doc_prefix + self.doc_suffix, add_special_tokens=False))
        budget = self.max_chunk_tokens - query_tokens - self.SPECIAL_TOKEN_BUDGET - doc_affix_overhead
        if budget <= 0:
            # Query alone fills the window; give the document at least half.
            budget = max(self.max_chunk_tokens // 2, 1)

        chunks: list[list[int]] = []
        current: list[int] = []
        current_tokens = 0
        for idx, (s_start, s_end) in enumerate(sentence_ranges):
            sent_tokens = len(self._tokenizer.encode(doc[s_start:s_end], add_special_tokens=False))
            if current and current_tokens + sent_tokens > budget:
                chunks.append(current)
                current = []
                current_tokens = 0
            current.append(idx)
            current_tokens += sent_tokens
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _build_sentence_spans(
        text: str,
        sentence_ranges: list[tuple[int, int]],
        kept: list[bool],
        sentence_scores: list[float],
    ) -> list[dict]:
        """Build contiguous ``{start, end, selected, score}`` spans at sentence boundaries."""
        spans: list[dict] = []
        pos = 0
        for (s_start, s_end), is_kept, score in zip(sentence_ranges, kept, sentence_scores):
            if pos < s_start:
                spans.append({"start": pos, "end": s_start, "selected": False, "score": 0.0})
            spans.append({"start": s_start, "end": s_end, "selected": is_kept, "score": float(score)})
            pos = s_end
        if pos < len(text):
            spans.append({"start": pos, "end": len(text), "selected": False, "score": 0.0})
        return spans

    @staticmethod
    def _sentence_scores_vectorized(
        token_starts: torch.Tensor,
        token_ends: torch.Tensor,
        token_probs: torch.Tensor,
        sent_starts: torch.Tensor,
        sent_ends: torch.Tensor,
    ) -> list[float]:
        """Compute per-sentence mean keep-probability via broadcasting.

        All tensors are 1-D. Returns a plain list of floats (one per sentence).
        """
        if token_starts.numel() == 0 or sent_starts.numel() == 0:
            return [0.0] * sent_starts.numel()
        # overlap: (S, T)
        overlap = (token_ends[None, :] > sent_starts[:, None]) & (token_starts[None, :] < sent_ends[:, None])
        counts = overlap.sum(dim=1).float()
        weighted = (overlap.float() * token_probs[None, :]).sum(dim=1)
        scores = torch.where(counts > 0, weighted / counts, torch.zeros_like(weighted))
        return scores.tolist()

    def _process_chunked_batch(
        self,
        query: str,
        docs: list[str],
        threshold: float,
        compute_spans: bool,
        batch_size: int,
    ) -> list[dict]:
        """Chunked + flat-batched inference for ``len(docs)`` documents.

        All chunks across all valid documents are forwarded together
        (sub-batched by ``batch_size``); per-sentence scores are stitched
        back into each document's original order, and the document-level
        ranking score is the maximum across its chunks.
        """
        n = len(docs)
        results: list[dict | None] = [None] * n
        valid_indices: list[int] = []
        valid_docs: list[str] = []
        for i, doc in enumerate(docs):
            if not doc or not isinstance(doc, str) or not doc.strip():
                empty: dict[str, Any] = {
                    "pruned_context": "",
                    "reranking_score": -999.0,
                    "compression_rate": 0.0,
                }
                if compute_spans:
                    empty["spans"] = [{"start": 0, "end": len(doc or ""), "selected": False}]
                results[i] = empty
            else:
                valid_indices.append(i)
                valid_docs.append(doc)

        if not valid_docs:
            return results  # type: ignore[return-value]

        # 1) Sentence-split all docs in one SaT call.
        all_sentence_ranges = self._sentence_ranges_batch(valid_docs)
        for j, ranges in enumerate(all_sentence_ranges):
            if not ranges:
                all_sentence_ranges[j] = [(0, len(valid_docs[j]))]

        # 2) Plan chunks per doc; flatten across docs into one list.
        flat_doc_idx: list[int] = []
        flat_sent_indices: list[list[int]] = []
        flat_chunk_texts: list[str] = []
        flat_chunk_starts: list[int] = []
        for di, doc in enumerate(valid_docs):
            ranges = all_sentence_ranges[di]
            chunks = self._plan_chunks(query, doc, ranges)
            for sent_indices in chunks:
                cs = ranges[sent_indices[0]][0]
                ce = ranges[sent_indices[-1]][1]
                flat_doc_idx.append(di)
                flat_sent_indices.append(sent_indices)
                flat_chunk_texts.append(doc[cs:ce])
                flat_chunk_starts.append(cs)

        # 3) Tokenize all (query, chunk) pairs at once with padding.
        prefix_len = len(self.doc_prefix)
        encoded = self._tokenizer(
            [self.query_prefix + query] * len(flat_chunk_texts),
            [self.doc_prefix + chunk + self.doc_suffix for chunk in flat_chunk_texts],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_chunk_tokens,
            padding=True,
            return_offsets_mapping=True,
        )
        offset_mappings = encoded.pop("offset_mapping")  # (C, L, 2)
        input_ids = encoded["input_ids"]  # (C, L)
        attention_masks = encoded["attention_mask"]  # (C, L)

        # 4) Sub-batched model forward over all chunks.
        effective_bs = self._chunk_batch(batch_size if batch_size > 0 else len(flat_chunk_texts))
        ranking_parts: list[torch.Tensor] = []
        compression_parts: list[torch.Tensor] = []
        for start in range(0, len(flat_chunk_texts), effective_bs):
            end = min(start + effective_bs, len(flat_chunk_texts))
            with torch.inference_mode():
                out = self.model(
                    input_ids=input_ids[start:end].to(self.device),
                    attention_mask=attention_masks[start:end].to(self.device),
                )
            ranking_parts.append(out["ranking_logits"].cpu())
            compression_parts.append(out["compression_logits"].cpu())

        ranking_logits = torch.cat(ranking_parts, dim=0)  # (C,)
        compression_logits = torch.cat(compression_parts, dim=0)  # (C, L, 2)
        keep_probs_all = torch.softmax(compression_logits, dim=-1)[:, :, 1]  # (C, L)

        # 5) Per-chunk: vectorized sentence scoring; aggregate into per-doc state.
        per_doc_scores: list[list[float]] = [[0.0] * len(all_sentence_ranges[di]) for di in range(len(valid_docs))]
        per_doc_ranking: list[list[float]] = [[] for _ in range(len(valid_docs))]

        for cidx, (di, sent_indices, chunk_start) in enumerate(zip(flat_doc_idx, flat_sent_indices, flat_chunk_starts)):
            per_doc_ranking[di].append(float(ranking_logits[cidx].item()))

            chunk_text = flat_chunk_texts[cidx]
            chunk_len = len(chunk_text)

            doc_start = self._find_doc_start_token(encoded, cidx)
            mask_slice = attention_masks[cidx, doc_start:]
            offsets_slice = offset_mappings[cidx, doc_start:]
            probs_slice = keep_probs_all[cidx, doc_start:]

            valid_tok = (mask_slice == 1) & ~((offsets_slice[:, 0] == 0) & (offsets_slice[:, 1] == 0))
            tok_starts_raw = offsets_slice[valid_tok, 0].float()
            tok_ends_raw = offsets_slice[valid_tok, 1].float()
            tok_probs = probs_slice[valid_tok]
            tok_starts = tok_starts_raw - prefix_len
            tok_ends = tok_ends_raw - prefix_len
            overlap_chunk = (tok_ends > 0) & (tok_starts < float(chunk_len))
            tok_starts = tok_starts[overlap_chunk].clamp(min=0, max=float(chunk_len))
            tok_ends = tok_ends[overlap_chunk].clamp(min=0, max=float(chunk_len))
            tok_probs = tok_probs[overlap_chunk]

            # Sentence ranges relative to this chunk's start.
            ranges_doc = all_sentence_ranges[di]
            sent_s = torch.tensor(
                [ranges_doc[si][0] - chunk_start for si in sent_indices],
                dtype=torch.float32,
            )
            sent_e = torch.tensor(
                [ranges_doc[si][1] - chunk_start for si in sent_indices],
                dtype=torch.float32,
            )

            chunk_sent_scores = self._sentence_scores_vectorized(
                tok_starts,
                tok_ends,
                tok_probs,
                sent_s,
                sent_e,
            )
            for local_idx, sent_idx in enumerate(sent_indices):
                per_doc_scores[di][sent_idx] = chunk_sent_scores[local_idx]

        # 6) Per-doc finalize: threshold, build pruned context + spans.
        for di, doc in enumerate(valid_docs):
            sentence_ranges = all_sentence_ranges[di]
            sentence_scores = per_doc_scores[di]
            ranking_score = max(per_doc_ranking[di]) if per_doc_ranking[di] else -999.0

            kept = [score >= threshold for score in sentence_scores]
            kept_sentences = [doc[s:e] for (s, e), k in zip(sentence_ranges, kept) if k]
            pruned_context = " ".join(kept_sentences)
            orig_len = len(doc)
            compression_rate = 100.0 * (1.0 - len(pruned_context) / orig_len) if orig_len else 0.0

            result: dict[str, Any] = {
                "pruned_context": pruned_context,
                "reranking_score": ranking_score,
                "compression_rate": compression_rate,
            }
            if compute_spans:
                result["spans"] = self._build_sentence_spans(
                    doc,
                    sentence_ranges,
                    kept,
                    sentence_scores,
                )
            results[valid_indices[di]] = result

        logger.info(
            "[CoCompress] batch | docs=%d  chunks=%d  threshold=%.3f",
            len(valid_docs),
            len(flat_chunk_texts),
            threshold,
        )
        return results  # type: ignore[return-value]

    def process_with_highlights(
        self,
        query: str,
        context: str,
        threshold: float | None = None,
        compute_spans: bool = True,
    ) -> dict:
        """Single-document path. Multi-chunk docs still benefit from
        chunk-level batched forwards via ``_process_chunked_batch``.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        threshold = threshold if threshold is not None else self.threshold
        return self._process_chunked_batch(
            query,
            [context],
            threshold,
            compute_spans,
            batch_size=8,
        )[0]

    def process_batch_with_highlights(
        self,
        query: str,
        docs: list[str],
        threshold: float | None = None,
        batch_size: int = 8,
        compute_spans: bool = True,
    ) -> list[dict]:
        """Batched reranking + compression for all docs under one query.

        Each document is sentence-split and chunked independently; chunks
        across the batch are flattened into one padded forward pass
        (sub-batched by ``batch_size``).
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        threshold = threshold if threshold is not None else self.threshold
        if not docs:
            return []
        return self._process_chunked_batch(
            query,
            docs,
            threshold,
            compute_spans,
            batch_size=batch_size,
        )

    def process_document(self, query: str, document: str) -> dict[str, Any]:
        """Single-document convenience wrapper; never returns spans."""
        return self.process_with_highlights(query, document, compute_spans=False)
