# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Patch for Qwen3-VL to fix LoRA module filtering issue.

Issue: vLLM's get_mm_mapping() returns prefixes with "model." but actual module names
       in vLLM v1 don't have this prefix, causing visual modules to not be filtered
       from LoRA wrapping, leading to AssertionError in lora_shrink_op.py.

Solution: Monkey-patch Qwen3VLForConditionalGeneration.get_mm_mapping() to return
          correct prefixes that match actual module names.
"""

import logging
from vllm.model_executor.models.module_mapping import MultiModelKeys

logger = logging.getLogger(__name__)


def _patched_qwen3_vl_get_mm_mapping(self) -> MultiModelKeys:
    """
    Patched version of get_mm_mapping for Qwen3-VL.

    Returns module prefixes WITHOUT "model." prefix to match actual vLLM v1 module names.
    This ensures visual modules are properly filtered from LoRA wrapping.
    """
    return MultiModelKeys.from_string_field(
        language_model="language_model",
        connector="visual.merger",  # Changed from "model.visual.merger"
        tower_model="visual.",      # Changed from "model.visual."
    )


def apply_qwen3_vl_lora_patch():
    """Apply the Qwen3-VL LoRA filtering patch."""
    try:
        from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

        # Store original method
        if not hasattr(Qwen3VLForConditionalGeneration, '_original_get_mm_mapping'):
            Qwen3VLForConditionalGeneration._original_get_mm_mapping = (
                Qwen3VLForConditionalGeneration.get_mm_mapping
            )

        # Apply patch
        Qwen3VLForConditionalGeneration.get_mm_mapping = _patched_qwen3_vl_get_mm_mapping

        logger.info("Successfully applied Qwen3-VL LoRA filtering patch")
        logger.info("Visual modules will now be properly excluded from LoRA wrapping")

    except ImportError as e:
        logger.warning(f"Could not apply Qwen3-VL patch: {e}")
        logger.warning("This is expected if not using Qwen3-VL or vLLM 0.11.0")


def remove_qwen3_vl_lora_patch():
    """Remove the Qwen3-VL LoRA filtering patch."""
    try:
        from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

        if hasattr(Qwen3VLForConditionalGeneration, '_original_get_mm_mapping'):
            Qwen3VLForConditionalGeneration.get_mm_mapping = (
                Qwen3VLForConditionalGeneration._original_get_mm_mapping
            )
            delattr(Qwen3VLForConditionalGeneration, '_original_get_mm_mapping')
            logger.info("Removed Qwen3-VL LoRA filtering patch")

    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not remove Qwen3-VL patch: {e}")
