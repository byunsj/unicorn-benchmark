"""Answer generation from compressed context.

The generator is deliberately plain: greedy decoding, a short answer budget, and
a prompt that forbids drawing on anything outside the context. Anything cleverer
would let the model paper over information the compressor discarded, which is
exactly the failure the benchmark is trying to observe.

Comp-S is the exception, and :class:`CompSGenerator` is here for it: it compresses
into memory embeddings that only its own decoder can read, so it answers with that
decoder instead of the shared model. The prompt and the decoding settings are the
same either way, so the two remain comparable.
"""

from __future__ import annotations

import re

from .llm import VLLMEngine

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"

SYSTEM_PROMPT = (
    "Answer the Question, using only information provided in the Context. "
    "If no useful information is provided, you must output 'No answer'."
)

PROMPT_TEMPLATE = "Context:\n{context}\n\nQuestion:\n{query}"


class Generator:
    """Generate answers with vLLM, one batch per call."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        max_new_tokens: int = 256,
        max_model_len: int | None = None,
        **engine_kwargs,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.engine = VLLMEngine(model_name, max_model_len=max_model_len, **engine_kwargs)

    def _prompt(self, query: str, context: str) -> str | dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(context=context or "No context.", query=query),
            },
        ]
        return self.engine.truncate(self.engine.chat_prompt(messages), self.max_new_tokens)

    def generate(self, queries: list[str], contexts: list[str]) -> list[dict]:
        """Answer each query from its context, in order."""
        from vllm import SamplingParams

        prompts = [self._prompt(q, c) for q, c in zip(queries, contexts)]
        params = SamplingParams(max_tokens=self.max_new_tokens, temperature=0.0)
        outputs = self.engine.generate(prompts, params, desc="Generate")
        return [
            {
                "generated_answer": out.outputs[0].text.strip(),
                "generator_input_tokens": len(out.prompt_token_ids),
                "generator_output_tokens": len(out.outputs[0].token_ids),
            }
            for out in outputs
        ]

    def unload(self) -> None:
        self.engine.unload()


# Matches the per-document headers build_context writes, so the joined context can
# be split back into the documents Comp-S has to compress one at a time.
_DOC_HEADER_RE = re.compile(r"##\s*Reference Document \(ID:.*?\)\n")


def _split_context(context: str) -> list[str]:
    if not context or not context.strip() or context.strip() == "No context.":
        return []
    docs = [part.strip() for part in _DOC_HEADER_RE.split(context) if part and part.strip()]
    return docs or [context.strip()]


def _escape_markers(text: str) -> str:
    """Neutralise literal ``<MEM>`` / ``<AE>`` in the input, as training did."""
    return text.replace("<MEM>", "< MEM >").replace("<AE>", "< AE >")


class CompSGenerator:
    """Generate answers by compressing documents to ``<MEM>`` embeddings, then decoding.

    The rows fed to the compressor and the ``<MEM>`` blocks written into the decoder
    prompt must line up exactly, so both are built from the same chunking helper the
    training collator used. A mismatch does not degrade quality gracefully: the
    memory embeddings land on the wrong positions.
    """

    # Slack subtracted from the chunk budget so the query prefix, the
    # re-tokenization boundary and the trailing markers still fit the window.
    SPECIAL_TOKEN_MARGIN = 8

    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int = 256,
        compressor_max_length: int = 128,
        max_chunks_per_doc: int = 16,
        decoder_max_length: int = 2048,
        batch_size: int = 8,
        # None picks FlashAttention-2 when it is installed and eager otherwise.
        attn_implementation: str | None = None,
    ):
        import torch

        from .models.soft_compression import SoftCompressionEncoder

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.compressor_max_length = compressor_max_length
        self.max_chunks_per_doc = max_chunks_per_doc
        self.decoder_max_length = decoder_max_length
        # Queries decoded per forward. Only the decoder is batched; compression runs
        # per query so its embeddings match single-query inference exactly.
        self.batch_size = max(1, batch_size)

        model = SoftCompressionEncoder.from_pretrained(model_name, attn_implementation=attn_implementation)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.query_dependent = bool(model.config.query_dependent)

    def _compressor_rows(self, query: str, docs: list[str]) -> tuple[list[list[int]], list[int]]:
        """Chunk the documents into ``<MEM>``-marked rows, plus each document's MEM total."""
        from .models.chunking import add_memory_tokens_to_inputs, chunk_doc_to_rows

        tokenizer = self.model.compressor_tokenizer
        rows: list[list[int]] = []
        chunks_per_doc: list[int] = []
        for doc in docs:
            doc_rows = chunk_doc_to_rows(
                tokenizer,
                query,
                doc,
                compressor_max_length=self.compressor_max_length,
                query_dependent=self.query_dependent,
                max_chunks_per_doc=self.max_chunks_per_doc,
                special_token_margin=self.SPECIAL_TOKEN_MARGIN,
            )
            rows.extend(doc_rows)
            chunks_per_doc.append(len(doc_rows))

        # BidirLM drops the representation of its last input position, so a trailing
        # token is appended there to protect the final <MEM> hidden state.
        bos_offset = self.model.config.compressor_bos_offset or 0
        rows, per_row = add_memory_tokens_to_inputs(
            rows,
            mem_token_id=tokenizer.mem_token_id,
            compr_rate=self.model.compr_rate,
            append_token_id=tokenizer.eos_token_id if bos_offset else None,
        )

        mems_per_doc: list[int] = []
        start = 0
        for n_chunks in chunks_per_doc:
            mems_per_doc.append(sum(per_row[start : start + n_chunks]))
            start += n_chunks
        return rows, mems_per_doc

    def _prompt_ids(self, query: str, mems_per_doc: list[int]) -> list[int]:
        tokenizer = self.model.decoder_tokenizer
        if mems_per_doc:
            context = "".join(
                f"Document {i}:" + tokenizer.mem_token * n for i, n in enumerate(mems_per_doc)
            )
        else:
            context = "No context."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_TEMPLATE.format(context=context, query=query)},
        ]
        # enable_thinking=False so the prompt ends with a closed empty think block,
        # matching how the training collator rendered the assistant turn. With an
        # open one the finetuned model emits an unbounded reasoning trace instead of
        # the short answer it was trained on.
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return tokenizer(
            text, add_special_tokens=False, truncation=True, max_length=self.decoder_max_length
        )["input_ids"]

    def _generate_batch(self, queries: list[str], contexts: list[str]) -> list[dict]:
        import torch

        compressor_tokenizer = self.model.compressor_tokenizer
        decoder_tokenizer = self.model.decoder_tokenizer

        embeddings: list = []
        prompts: list[list[int]] = []
        for raw_query, raw_context in zip(queries, contexts):
            query = _escape_markers(raw_query or "")
            docs = [_escape_markers(d) for d in _split_context(raw_context or "")]
            mems_per_doc: list[int] = []
            if docs:
                rows, mems_per_doc = self._compressor_rows(query, docs)
                batch = compressor_tokenizer.pad({"input_ids": rows}, padding="longest", return_tensors="pt")
                with torch.inference_mode():
                    embeddings.append(
                        self.model.compress(
                            batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)
                        )
                    )
            prompts.append(self._prompt_ids(query, mems_per_doc))

        # The decoder tokenizer pads on the left, so the <MEM> positions of the
        # padded batch stay in query order and match the concatenated embeddings.
        padded = decoder_tokenizer.pad({"input_ids": prompts}, padding="longest", return_tensors="pt")
        input_ids = padded["input_ids"].to(self.device)
        attention_mask = padded["attention_mask"].to(self.device)

        with torch.inference_mode():
            if embeddings:
                memory = torch.cat(embeddings, dim=0)
                n_slots = int((input_ids == decoder_tokenizer.mem_token_id).sum().item())
                if n_slots != memory.shape[0]:
                    raise ValueError(
                        f"decoder has {n_slots} <MEM> slots but {memory.shape[0]} memory embeddings; "
                        f"the prompt was truncated at decoder_max_length={self.decoder_max_length}. "
                        f"Lower max_chunks_per_doc (now {self.max_chunks_per_doc}) or raise it."
                    )
                output_ids = self.model.decoder.generate(
                    inputs_embeds=self.model.replace_embeddings(memory, input_ids),
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=decoder_tokenizer.eos_token_id,
                    pad_token_id=decoder_tokenizer.pad_token_id,
                )
            else:
                output_ids = self.model.decoder.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=decoder_tokenizer.eos_token_id,
                    pad_token_id=decoder_tokenizer.pad_token_id,
                )[:, input_ids.shape[1] :]

        answers = decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [
            {
                "generated_answer": (answer or "").strip(),
                "generator_input_tokens": len(prompt),
                "generator_output_tokens": len(ids),
            }
            for answer, prompt, ids in zip(answers, prompts, output_ids)
        ]

    def generate(self, queries: list[str], contexts: list[str]) -> list[dict]:
        """Answer each query from its context, in order."""
        results: list[dict] = []
        for start in range(0, len(queries), self.batch_size):
            stop = start + self.batch_size
            results.extend(self._generate_batch(queries[start:stop], contexts[start:stop]))
        return results

    def unload(self) -> None:
        import gc

        import torch

        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
