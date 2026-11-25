"""Model architectures vendored so released checkpoints load without remote code.

``bidirlm`` and ``qwen3_5`` are public architectures that the pinned transformers
version does not ship, so importing them here registers both with the ``Auto*``
classes. Keeping the pin is deliberate: it is the version the reported numbers
were produced with.
"""

from . import bidirlm

__all__ = ["bidirlm"]
