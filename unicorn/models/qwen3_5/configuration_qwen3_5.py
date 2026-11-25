"""Qwen3.5 configuration, written in the transformers 4.57 ``__init__`` style.

The upstream ``configuration_qwen3_5.py`` uses the transformers-5 ``@strict``
dataclass config style (``PreTrainedConfig`` with typed class attributes), which
is not supported by the 4.57.x pinned here. This module reimplements the same
three config classes (text / vision / top-level) using the classic
``PretrainedConfig.__init__`` pattern, accepting exactly the fields present in
the ``Qwen/Qwen3.5-0.8B`` ``config.json``.
"""

from transformers.configuration_utils import PretrainedConfig


class Qwen3_5TextConfig(PretrainedConfig):
    model_type = "qwen3_5_text"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        attn_output_gate=True,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        layer_types=None,
        full_attention_interval=4,
        mlp_only_layers=None,
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        mamba_ssm_dtype="float32",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attn_output_gate = attn_output_gate

        # Linear-attention (Gated DeltaNet) dims.
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # Hybrid layer schedule: every ``full_attention_interval``-th layer is a
        # full-attention layer, the rest are linear-attention layers.
        self.full_attention_interval = full_attention_interval
        if layer_types is None:
            layer_types = [
                "full_attention" if (i + 1) % full_attention_interval == 0 else "linear_attention"
                for i in range(num_hidden_layers)
            ]
        self.layer_types = layer_types
        self.mlp_only_layers = mlp_only_layers if mlp_only_layers is not None else []

        # Multi-token-prediction head metadata (weights are ignored at load time).
        self.mtp_num_hidden_layers = mtp_num_hidden_layers
        self.mtp_use_dedicated_embeddings = mtp_use_dedicated_embeddings
        self.mamba_ssm_dtype = mamba_ssm_dtype

        # The Qwen3.5 rotary embedding reads ``config.rope_parameters`` as a dict
        # (mrope), unlike the 4.57 ``rope_theta``/``rope_scaling`` split.
        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            }
        self.rope_parameters = rope_parameters

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3_5VisionConfig(PretrainedConfig):
    model_type = "qwen3_5_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        deepstack_visual_indexes=None,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.deepstack_visual_indexes = deepstack_visual_indexes if deepstack_visual_indexes is not None else []
        self.initializer_range = initializer_range


class Qwen3_5Config(PretrainedConfig):
    model_type = "qwen3_5"
    sub_configs = {"vision_config": Qwen3_5VisionConfig, "text_config": Qwen3_5TextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            # Some checkpoints ship the vision sub-config with the top-level model
            # type ("qwen3_5") rather than "qwen3_5_vision"; normalize it.
            if vision_config.get("model_type") == "qwen3_5":
                vision_config = {**vision_config, "model_type": "qwen3_5_vision"}
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


__all__ = ["Qwen3_5Config", "Qwen3_5TextConfig", "Qwen3_5VisionConfig"]
