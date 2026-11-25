"""BidirLM: a Qwen3-based bidirectional encoder, used as the Comp-H backbone.

Vendored here (config, encoder, and the sequence-classification head with its
pooling) so ``CompressionEncoder`` can rebuild the backbone through
``AutoModelForSequenceClassification`` without executing remote code.

Importing this module registers ``bidirlm`` with ``AutoConfig`` /
``AutoModelForSequenceClassification``.
"""


import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update, rope_config_validation
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

try:
    import flash_attn

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

logger = logging.get_logger(__name__)


class BidirLMConfig(PretrainedConfig):
    r"""Configuration for :class:`BidirLMModel` (a Qwen3-style bidirectional encoder)."""

    model_type = "bidirlm"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    @property
    def effective_sliding_window(self):
        sw = getattr(self, "sliding_window", None)
        if sw is None:
            return None
        if getattr(self, "use_bidirectional_attention", False):
            return sw // 2
        return sw

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        classifier_pooling="late",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.clf_pooling = classifier_pooling
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """Qwen3RMSNorm is equivalent to T5LayerNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, None, :, :].expand(num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(num_key_value_heads * n_rep, slen, head_dim)


def batch_input_to_cu_seqlens(x: torch.Tensor, attention_mask: torch.Tensor):
    lengths = attention_mask.sum(dim=1)
    max_seqlen = int(lengths.max().item())
    cu_seqlens = torch.zeros(lengths.size(0) + 1, dtype=torch.int32, device=x.device)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    x = x[attention_mask.bool()]
    return x, cu_seqlens, max_seqlen


def cu_seqlens_to_batch_input(x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int):
    B = cu_seqlens.size(0) - 1
    D = x.size(1)
    idx = torch.arange(max_seqlen, device=x.device).expand(B, max_seqlen)
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).unsqueeze(1)
    mask = idx < lens
    base = cu_seqlens[:-1].unsqueeze(1)
    gather_idx = (idx + base) * mask
    out = torch.zeros(B, max_seqlen, D, device=x.device, dtype=x.dtype)
    out[mask] = x[gather_idx[mask]]
    return out


def cu_attention_weight_to_batch(hidden_states, cu_seqlens, max_seqlen):
    H, T, _ = hidden_states.shape
    device = hidden_states.device
    cu_seqlens = cu_seqlens.to(device, dtype=torch.long)

    B = cu_seqlens.numel() - 1
    start = cu_seqlens[:-1]
    end = cu_seqlens[1:]
    L = end - start

    p = torch.arange(max_seqlen, device=device)
    valid = p.unsqueeze(0) < L.unsqueeze(1)

    rel = p.unsqueeze(0)
    abs_idx = start.unsqueeze(1) + rel
    abs_idx = torch.where(valid, abs_idx, torch.zeros_like(abs_idx))

    attn = hidden_states.unsqueeze(0).expand(B, -1, -1, -1)

    row_index = abs_idx[:, None, :, None].expand(B, H, max_seqlen, T)
    attn_rows = torch.gather(attn, dim=2, index=row_index)

    col_index = abs_idx[:, None, None, :].expand(B, H, max_seqlen, max_seqlen)
    attn_padded = torch.gather(attn_rows, dim=3, index=col_index)

    mask = valid.to(attn_padded.dtype)
    attn_padded = attn_padded * mask[:, None, :, None] * mask[:, None, None, :]

    return attn_padded


def create_packed_seqs_mask(
    cu_seqlens: torch.Tensor,
    causal: bool = True,
    device: torch.device = torch.device("cpu"),  # noqa: B008
) -> torch.Tensor:
    """Create a causal or non-causal attention mask for packed sequences."""
    total_len = cu_seqlens[-1].item()
    seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]

    seq_indices = torch.repeat_interleave(
        torch.arange(len(seq_lengths), device=device),
        seq_lengths,
    )

    seq_mask = seq_indices.unsqueeze(0) == seq_indices.unsqueeze(1)

    if causal:
        causal_mask = torch.tril(torch.ones(total_len, total_len, device=device, dtype=torch.bool))
        combined_mask = seq_mask & causal_mask
    else:
        combined_mask = seq_mask

    attention_mask = torch.full((total_len, total_len), float("-inf"), device=device)
    attention_mask.masked_fill_(combined_mask, 0.0)

    return attention_mask


def sdpa_attention_forward(
    q,
    k,
    v,
    cu_seqlens,
    scaling,
    dropout: float = 0.0,
    causal: bool = True,
):
    """Compute scaled dot-product attention for packed sequences."""
    attn_weights = torch.matmul(q, k.transpose(1, 2)) * scaling

    mask = create_packed_seqs_mask(cu_seqlens, causal, q.device)
    attn_weights = attn_weights + mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous()

    return attn_output, attn_weights


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: BidirLMConfig):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(0, 1)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(0, 1)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(0, 1)

        query_states, key_states = query_states.unsqueeze(0), key_states.unsqueeze(0)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states, key_states = query_states.squeeze(0), key_states.squeeze(0)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.config._attn_implementation == "flash_attention_2":
            attn_weights = None
            attn_output = flash_attn.flash_attn_varlen_func(
                query_states.transpose(0, 1),
                key_states.transpose(0, 1),
                value_states.transpose(0, 1),
                cu_seqlens,
                cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=self.scaling,
                causal=False,
            ).contiguous()
        else:
            attn_output, attn_weights = sdpa_attention_forward(
                query_states,
                key_states,
                value_states,
                cu_seqlens=cu_seqlens,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling,
                causal=False,
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Qwen3EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: BidirLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # necessary, but kept here for BC
        output_attentions: bool | None = False,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class BidirLMPreTrainedModel(PreTrainedModel):
    config: BidirLMConfig
    base_model_prefix = "model"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_record_outputs = {}


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: BidirLMConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seqlen_cached = config.max_position_embeddings
        self.original_max_seqlen = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class BidirLMModel(BidirLMPreTrainedModel):
    def __init__(self, config: BidirLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen3EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.mask_converter = AttentionMaskConverter(True)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        *,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | BaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # For MNTP XP
        batch_size, seq_len = input_ids.size()
        new_input_ids = torch.empty((batch_size, seq_len + 1), dtype=input_ids.dtype, device=input_ids.device)
        new_input_ids[:, 0] = 151644
        new_input_ids[:, 1:] = input_ids

        if attention_mask is not None:
            new_attention_mask = torch.empty(
                (batch_size, seq_len + 1), dtype=attention_mask.dtype, device=attention_mask.device
            )
            new_attention_mask[:, 0] = 1
            new_attention_mask[:, 1:] = attention_mask
            attention_mask = new_attention_mask
            input_ids, cu_seqlens, max_seqlen = batch_input_to_cu_seqlens(new_input_ids, attention_mask)
        else:
            input_ids = new_input_ids

        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(len(input_ids), device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                if attention_mask is not None:
                    # cu_seqlens_to_batch_input returns a 3-D (B, L, D) tensor; the
                    # trailing [0] in upstream wrongly collapsed it to 2-D, which breaks
                    # the ``h[:, :-1, :]`` slicing below whenever output_hidden_states=True.
                    all_hidden_states += (
                        cu_seqlens_to_batch_input(hidden_states, cu_seqlens, attention_mask.shape[-1]),
                    )
                else:
                    all_hidden_states += (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                if attention_mask is not None:
                    all_self_attns += (
                        cu_attention_weight_to_batch(layer_outputs[1], cu_seqlens, attention_mask.shape[-1]),
                    )
                else:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if attention_mask is not None:
            hidden_states = cu_seqlens_to_batch_input(hidden_states, cu_seqlens, attention_mask.shape[-1])
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # For MNTP XP
        output = BaseModelOutput(
            last_hidden_state=hidden_states[:, :-1, :],
            hidden_states=tuple(h[:, :-1, :] for h in all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(a[:, :, :-1, :-1] for a in all_self_attns) if all_self_attns is not None else None,
        )
        return output if return_dict else output.to_tuple()


def clf_pool_and_project(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor | None,
    clf_pooling: str,
    dense: nn.Module,
    activation: nn.Module,
    classifier: nn.Module,
) -> torch.Tensor:
    """Pool + project exactly like ``BidirLMForSequenceClassification``.

    Identical to the pooling used at training time, so ranking scores reproduce:

    - ``"bos"``: take position 0, then ``classifier(activation(dense(x)))``.
    - ``"mean"``: masked mean over valid tokens, then ``classifier(activation(dense(x)))``.
    - ``"late"``: apply ``classifier(activation(dense(x)))`` per token, then masked mean.

    Note that ``"late"`` differs from ``"mean"`` because of the ``activation`` non-linearity.
    """
    if clf_pooling in ("bos", "mean"):
        if clf_pooling == "bos":
            pooled = last_hidden_state[:, 0]
        elif attention_mask is None:
            pooled = last_hidden_state.mean(dim=1)
        else:
            pooled = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / attention_mask.sum(dim=1, keepdim=True)
        return classifier(activation(dense(pooled)))
    if clf_pooling == "late":
        logits = classifier(activation(dense(last_hidden_state)))
        if attention_mask is None:
            return logits.mean(dim=1)
        logits = (logits * attention_mask.unsqueeze(-1)).sum(dim=1)
        return logits / attention_mask.sum(dim=1, keepdim=True)
    raise ValueError(f"Unsupported clf_pooling: {clf_pooling!r} (expected 'bos', 'mean', or 'late')")


class BidirLMForSequenceClassification(BidirLMPreTrainedModel):
    def __init__(self, config: BidirLMConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.clf_pooling = config.clf_pooling

        self.model = BidirLMModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.post_init()

    def compute_pooled_logits(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool ``last_hidden_state`` into ``[B, num_labels]`` scores via ``clf_pooling``.

        Lets ``_CompressionEncoder`` reproduce training's ranking exactly by running the
        backbone once and reading ``last_hidden_state`` (mirrors ``BidirLMForReranking``
        in the training package), instead of requesting ``output_hidden_states``.
        """
        return clf_pool_and_project(
            last_hidden_state, attention_mask, self.clf_pooling, self.dense, self.activation, self.classifier
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_output = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_output[0]

        logits = clf_pool_and_project(
            last_hidden_state, attention_mask, self.clf_pooling, self.dense, self.activation, self.classifier
        )

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        output = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )
        return output if return_dict else output.to_tuple()


AutoConfig.register("bidirlm", BidirLMConfig, exist_ok=True)
AutoModel.register(BidirLMConfig, BidirLMModel, exist_ok=True)
AutoModelForSequenceClassification.register(BidirLMConfig, BidirLMForSequenceClassification, exist_ok=True)


__all__ = [
    "BidirLMConfig",
    "BidirLMForSequenceClassification",
    "BidirLMModel",
    "BidirLMPreTrainedModel",
    "clf_pool_and_project",
]
