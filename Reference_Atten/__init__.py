"""SD3 reference-attention processors; importing this package loads no other backbone."""

from .sd3_reference_attn_processor import (
    SD3ReferenceReadJointAttnProcessor2_0,
    SD3ReferenceWriteJointAttnProcessor2_0,
    set_sd3_reference_read_attn,
    set_sd3_reference_write_attn,
)

__all__ = [
    "SD3ReferenceReadJointAttnProcessor2_0",
    "SD3ReferenceWriteJointAttnProcessor2_0",
    "set_sd3_reference_read_attn",
    "set_sd3_reference_write_attn",
]
