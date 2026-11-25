"""Vendored Qwen3.5 (hybrid Gated-DeltaNet + Qwen3-VL vision) model.

Copied from transformers ``main`` and patched to run under the
``transformers ~= 4.57`` pinned by this project. See ``modeling_qwen3_5.py`` and
``_compat.py`` for details on the (minimal) edits.
"""

from .configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from .modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5PreTrainedModel,
    Qwen3_5TextModel,
    Qwen3_5VisionModel,
)

__all__ = [
    "Qwen3_5Config",
    "Qwen3_5TextConfig",
    "Qwen3_5VisionConfig",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5Model",
    "Qwen3_5TextModel",
    "Qwen3_5VisionModel",
    "Qwen3_5PreTrainedModel",
    "register_qwen3_5",
]

_registered = False


def register_qwen3_5(*, exist_ok: bool = True) -> None:
    """Register the vendored Qwen3.5 classes with the ``transformers`` ``Auto*`` APIs.

    Required even for plain instantiation: ``Qwen3_5Model.__init__`` builds its
    vision/text sub-models via ``AutoModel.from_config(...)``. Also wires the
    processor/image-processor (reused from Qwen3-VL / Qwen2-VL) and tokenizer so
    ``AutoProcessor.from_pretrained`` works on a Qwen3.5 checkpoint. Idempotent.
    """
    global _registered
    if _registered:
        return

    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModel,
        AutoModelForCausalLM,
        AutoProcessor,
        Qwen2Tokenizer,
        Qwen2TokenizerFast,
        Qwen2VLImageProcessorFast,
        Qwen3VLProcessor,
    )
    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

    # auto_docstring may leave ``config_class`` as a string on vendored classes;
    # AutoModel.register compares ``.__name__``, so ensure it is the real class.
    for model_cls, cfg_cls in (
        (Qwen3_5Model, Qwen3_5Config),
        (Qwen3_5TextModel, Qwen3_5TextConfig),
        (Qwen3_5VisionModel, Qwen3_5VisionConfig),
        (Qwen3_5ForConditionalGeneration, Qwen3_5Config),
    ):
        if not isinstance(getattr(model_cls, "config_class", None), type):
            model_cls.config_class = cfg_cls

    AutoConfig.register(Qwen3_5Config.model_type, Qwen3_5Config, exist_ok=exist_ok)
    AutoConfig.register(Qwen3_5TextConfig.model_type, Qwen3_5TextConfig, exist_ok=exist_ok)
    AutoConfig.register(Qwen3_5VisionConfig.model_type, Qwen3_5VisionConfig, exist_ok=exist_ok)

    AutoModel.register(Qwen3_5VisionConfig, Qwen3_5VisionModel, exist_ok=exist_ok)
    AutoModel.register(Qwen3_5TextConfig, Qwen3_5TextModel, exist_ok=exist_ok)
    AutoModel.register(Qwen3_5Config, Qwen3_5Model, exist_ok=exist_ok)
    AutoModelForCausalLM.register(Qwen3_5Config, Qwen3_5ForConditionalGeneration, exist_ok=exist_ok)

    # Image-text-to-text is the natural head, but the auto class is not present in
    # every 4.57.x; register it opportunistically.
    try:
        from transformers import AutoModelForImageTextToText

        AutoModelForImageTextToText.register(Qwen3_5Config, Qwen3_5ForConditionalGeneration, exist_ok=exist_ok)
    except Exception:
        pass

    # Qwen3.5 reuses the Qwen3-VL processor + Qwen2-VL fast image processor + Qwen2 tokenizer.
    AutoImageProcessor.register(Qwen3_5Config, fast_image_processor_class=Qwen2VLImageProcessorFast, exist_ok=exist_ok)
    if Qwen3_5Config not in TOKENIZER_MAPPING:
        TOKENIZER_MAPPING.register(Qwen3_5Config, (Qwen2Tokenizer, Qwen2TokenizerFast), exist_ok=exist_ok)
    AutoProcessor.register(Qwen3_5Config, Qwen3VLProcessor, exist_ok=exist_ok)

    _registered = True


register_qwen3_5()
