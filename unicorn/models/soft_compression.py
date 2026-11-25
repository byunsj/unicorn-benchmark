"""Comp-S encoder: compress documents into memory embeddings, then decode.

Follows PISCO (https://arxiv.org/abs/2501.16075) and OSCAR
(https://arxiv.org/abs/2504.07109):

- a **compressor** reads a document (query-prefixed, since Comp-S is
  query-dependent) followed by ``<MEM>`` marker tokens, giving one hidden state
  per marker;
- a **connector** MLP maps those states into the decoder's embedding space;
- a **decoder** (LoRA-adapted) takes them in place of its ``<MEM>`` placeholders
  and generates the answer.

Because the compressed form is a set of embeddings rather than text, Comp-S
answers with its own decoder instead of handing text to an external generator.

This is the inference half of the training model: the pieces needed to load a
released checkpoint and generate. Training losses and checkpoint writing are not
included.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from .attention import select_attn_implementation

# Backbones whose forward prepends a BOS token and drops the last hidden state,
# so ``last_hidden_state[:, j]`` represents input token ``j - 1``.
_COMPRESSOR_BOS_OFFSET: dict[str, int] = {"bidirlm": 1}

MEM_TOKEN = "<MEM>"
AE_TOKEN = "<AE>"


class SoftCompressionConfig(PretrainedConfig):
    model_type = "soft_compression"

    def __init__(
        self,
        compressor_model_name: str = "BidirLM/BidirLM-0.6B-Base",
        decoder_model_name: str = "Qwen/Qwen3.5-9B",
        compressor_tokenizer_name: str | None = None,
        compr_rate: int = 16,
        compressor_mlp_hidden_dim: int = 8192,
        lora_decoder: bool = True,
        lora_r_decoder: int = 64,
        query_dependent: bool = True,
        attn_implementation: str = "flash_attention_2",
        load_decoder: bool = True,
        compressor_bos_offset: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.compressor_model_name = compressor_model_name
        self.decoder_model_name = decoder_model_name
        self.compressor_tokenizer_name = compressor_tokenizer_name or compressor_model_name
        self.compr_rate = compr_rate
        self.compressor_mlp_hidden_dim = compressor_mlp_hidden_dim
        self.lora_decoder = lora_decoder
        self.lora_r_decoder = lora_r_decoder
        self.query_dependent = query_dependent
        self.attn_implementation = attn_implementation
        self.load_decoder = load_decoder
        self.compressor_bos_offset = compressor_bos_offset


class SoftCompressionEncoder(PreTrainedModel):
    config_class = SoftCompressionConfig

    def __init__(self, config: SoftCompressionConfig):
        super().__init__(config)

        # Qwen3.5 is not a built-in architecture on the transformers pin this
        # benchmark uses, so register the vendored port before any Auto* call.
        from .qwen3_5 import register_qwen3_5

        register_qwen3_5()

        self.decoder_tokenizer = self._create_decoder_tokenizer(config)
        if config.load_decoder:
            self.decoder = self._create_decoder(config)

        decoder_config = AutoConfig.from_pretrained(config.decoder_model_name, trust_remote_code=True)
        decoder_hidden = self._resolve_hidden_size(decoder_config)

        self.compressor_tokenizer = self._create_compressor_tokenizer(config)
        self.compressor, self.connector = self._create_compressor_and_connector(config, decoder_hidden)

        if config.compressor_bos_offset is None:
            config.compressor_bos_offset = _COMPRESSOR_BOS_OFFSET.get(self.compressor.config.model_type, 0)
        self.compr_rate = config.compr_rate

    # ---------------------------------------------------------------- builders

    @staticmethod
    def _resolve_hidden_size(config: PretrainedConfig) -> int:
        if hasattr(config, "hidden_size"):
            return config.hidden_size
        text_config = getattr(config, "text_config", None)
        if text_config is not None and hasattr(text_config, "hidden_size"):
            return text_config.hidden_size
        raise ValueError(f"Cannot resolve hidden_size from config {type(config).__name__}")

    def _create_decoder(self, config: SoftCompressionConfig) -> PreTrainedModel:
        hf_config = AutoConfig.from_pretrained(config.decoder_model_name, trust_remote_code=True)
        dtype = torch.bfloat16

        if getattr(hf_config, "model_type", None) == "qwen3_5":
            # Load the full multimodal checkpoint so the text weights, stored under
            # ``model.language_model.*``, map correctly, then drop the unused vision
            # tower. Passing no pixels keeps the forward text-only.
            from .qwen3_5 import Qwen3_5ForConditionalGeneration

            hf_config._attn_implementation = config.attn_implementation
            decoder = Qwen3_5ForConditionalGeneration.from_pretrained(
                config.decoder_model_name, config=hf_config, dtype=dtype
            )
            if getattr(decoder.model, "visual", None) is not None:
                decoder.model.visual = None
        else:
            decoder = AutoModelForCausalLM.from_pretrained(
                config.decoder_model_name,
                attn_implementation=config.attn_implementation,
                dtype=dtype,
                trust_remote_code=True,
            )

        decoder.resize_token_embeddings(len(self.decoder_tokenizer))
        if config.lora_decoder:
            decoder.add_adapter(self._peft_config(config.lora_r_decoder))
        return decoder

    def _create_compressor_and_connector(
        self, config: SoftCompressionConfig, decoder_hidden_dim: int
    ) -> tuple[PreTrainedModel, nn.Module]:
        # Importing bidirlm registers it with the Auto* classes. Loading without
        # trust_remote_code is deliberate: the Hub copy's remote code needs
        # transformers>=5, so we use the vendored implementation and fetch only weights.
        from . import bidirlm  # noqa: F401

        compressor = AutoModel.from_pretrained(
            config.compressor_model_name,
            attn_implementation=config.attn_implementation,
            dtype=torch.bfloat16,
        )
        compressor.resize_token_embeddings(len(self.compressor_tokenizer))

        connector = nn.Sequential(
            nn.Linear(self._resolve_hidden_size(compressor.config), config.compressor_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.compressor_mlp_hidden_dim, decoder_hidden_dim),
        ).to(torch.bfloat16)
        return compressor, connector

    def _create_decoder_tokenizer(self, config: SoftCompressionConfig):
        tok = AutoTokenizer.from_pretrained(
            config.decoder_model_name, padding_side="left", truncation_side="right", trust_remote_code=True
        )
        tok.add_special_tokens({"additional_special_tokens": [MEM_TOKEN, AE_TOKEN]})
        tok.mem_token = MEM_TOKEN
        tok.mem_token_id = tok.convert_tokens_to_ids(MEM_TOKEN)
        tok.ae_token = AE_TOKEN
        tok.ae_token_id = tok.convert_tokens_to_ids(AE_TOKEN)
        tok.bos_token = tok.bos_token or ""
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        return tok

    def _create_compressor_tokenizer(self, config: SoftCompressionConfig):
        tok = AutoTokenizer.from_pretrained(
            config.compressor_tokenizer_name, padding_side="left", trust_remote_code=True
        )
        tok.add_special_tokens({"additional_special_tokens": [MEM_TOKEN]})
        tok.mem_token = MEM_TOKEN
        tok.mem_token_id = tok.convert_tokens_to_ids(MEM_TOKEN)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        return tok

    @staticmethod
    def _peft_config(lora_r: int) -> LoraConfig:
        return LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_r,
            lora_alpha=2 * lora_r,
            target_modules="all-linear",
            lora_dropout=0.1,
            modules_to_save=["embed_tokens", "lm_head"],
        )

    # ---------------------------------------------------------------- core ops

    def _align_hidden(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Realign hidden states so ``hidden[:, i]`` corresponds to token ``i``."""
        offset = self.config.compressor_bos_offset or 0
        if offset:
            hidden = hidden[:, offset:, :]
        return hidden, input_ids[:, : hidden.shape[1]]

    def _mem_embeddings(self, hidden: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        mem_mask = ids == self.compressor_tokenizer.mem_token_id
        all_hidden = torch.cat([hidden[i, mem_mask[i], :] for i in range(hidden.shape[0])], dim=0)

        connector_param = next(self.connector.parameters(), None)
        if connector_param is not None:
            all_hidden = all_hidden.to(connector_param.dtype)
        return self.connector(all_hidden)

    def compress(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compress ``<MEM>``-marked inputs into decoder-space embeddings.

        Returns ``(sum_i n_mem_i, decoder_hidden)``, concatenated in row-major order.
        """
        outputs = self.compressor(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        raw = outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
        hidden, ids = self._align_hidden(raw, input_ids)
        return self._mem_embeddings(hidden, ids)

    def replace_embeddings(self, compressed_embs: torch.Tensor, dec_input_ids: torch.Tensor) -> torch.Tensor:
        """Substitute compressed embeddings for the decoder's ``<MEM>`` placeholders."""
        dec_embeds = self.decoder.get_input_embeddings()(dec_input_ids)
        b, length, hidden_dim = dec_embeds.shape

        mem_mask = dec_input_ids == self.decoder_tokenizer.mem_token_id
        num_mem = int(mem_mask.sum().item())
        if num_mem != compressed_embs.shape[0]:
            raise ValueError(
                f"{num_mem} decoder <MEM> tokens but {compressed_embs.shape[0]} compressed embeddings"
            )

        mem_indices = mem_mask.view(-1).nonzero(as_tuple=False).squeeze(1)
        flat = dec_embeds.view(-1, hidden_dim)
        flat = flat.index_copy(0, mem_indices, compressed_embs.to(flat.dtype))
        return flat.view(b, length, hidden_dim)

    @torch.no_grad()
    def generate(self, model_input: dict[str, torch.Tensor], max_new_tokens: int = 128) -> list[str]:
        embeddings = self.compress(model_input["compressor_input_ids"], model_input["compressor_attention_mask"])
        dec_inputs_embeds = self.replace_embeddings(embeddings, model_input["decoder_input_ids"])
        # The base decoder's generation_config stops on <|endoftext|>, but finetuning
        # taught the chat stop token <|im_end|>. Pass it explicitly or generation
        # rambles into a hallucinated next turn.
        output_ids = self.decoder.generate(
            inputs_embeds=dec_inputs_embeds,
            attention_mask=model_input["decoder_attention_mask"],
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.decoder_tokenizer.eos_token_id,
            pad_token_id=self.decoder_tokenizer.pad_token_id,
        )
        return self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        load_decoder: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        """Load a released checkpoint from the Hub or a local directory.

        The pieces are stored separately rather than as one state dict, so they are
        reattached here: LoRA decoder state, the compressor, and the connector.
        """
        path = Path(pretrained_model_name_or_path)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(repo_id=pretrained_model_name_or_path))

        config = SoftCompressionConfig.from_pretrained(str(path))
        config.attn_implementation = select_attn_implementation(
            kwargs.get("attn_implementation", config.attn_implementation)
        )
        config.load_decoder = load_decoder

        model = cls(config)

        decoder_state = path / "decoder_state.pt"
        if load_decoder and decoder_state.is_file():
            model.decoder.load_state_dict(torch.load(decoder_state, map_location="cpu"), strict=False)

        compressor_dir = path / "compressor"
        if compressor_dir.is_dir():
            model.compressor = AutoModel.from_pretrained(
                str(compressor_dir), dtype=torch.bfloat16, attn_implementation=config.attn_implementation
            )

        connector_path = path / "connector.pt"
        if connector_path.is_file():
            model.connector.load_state_dict(torch.load(connector_path, map_location="cpu"))
            model.connector.to(torch.bfloat16)
        return model


__all__ = ["AE_TOKEN", "MEM_TOKEN", "SoftCompressionConfig", "SoftCompressionEncoder"]
