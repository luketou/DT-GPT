"""Unsloth integration helpers for DT-GPT.

Provides thin wrappers around ``unsloth.FastLanguageModel`` so that the rest
of the pipeline can swap between the standard PEFT path
(``pipeline.lora_helpers``) and the Unsloth-accelerated path with minimal
code changes.

Requires the ``unsloth`` package (Python >= 3.9).  All public functions in
this module raise ``ImportError`` with a helpful message when Unsloth is not
installed, so callers can gate on :func:`is_unsloth_available` first.
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_unsloth_available():
    """Return *True* if the ``unsloth`` package can be imported."""
    try:
        import unsloth  # noqa: F401
        return True
    except ImportError:
        return False


def _require_unsloth():
    if not is_unsloth_available():
        raise ImportError(
            "Unsloth is not installed in this environment.  "
            "Please activate the 'dtgpt-unsloth' conda environment or "
            "install unsloth: pip install unsloth"
        )


# ---------------------------------------------------------------------------
# Default target modules (same as lora_helpers for Mistral-family models)
# ---------------------------------------------------------------------------

DEFAULT_MISTRAL_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_unsloth(
    model_name,
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,
):
    """Load a model and tokenizer via Unsloth's ``FastLanguageModel``.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path (e.g. ``"BioMistral/BioMistral-7B-DARE"``).
    max_seq_length : int
        Maximum sequence length the model will see during training.
    load_in_4bit : bool
        If *True*, load with 4-bit NF4 quantisation (QLoRA).
    dtype : torch.dtype or None
        Compute dtype.  ``None`` lets Unsloth auto-detect
        (bfloat16 on Ampere+, float16 otherwise).

    Returns
    -------
    model, tokenizer
    """
    _require_unsloth()
    from unsloth import FastLanguageModel

    logger.info(
        "Loading model via Unsloth: %s  (4-bit=%s, max_seq=%d)",
        model_name,
        load_in_4bit,
        max_seq_length,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# PEFT / DoRA adapter
# ---------------------------------------------------------------------------

def apply_unsloth_peft(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=None,
    use_dora=True,
    use_gradient_checkpointing=True,
    random_state=42,
):
    """Wrap *model* with LoRA/DoRA adapters using Unsloth's optimised path.

    Parameters
    ----------
    model
        A model returned by :func:`load_model_unsloth`.
    r : int
        LoRA rank.
    lora_alpha : int
        LoRA alpha scaling factor.
    lora_dropout : float
        Dropout probability for LoRA layers.
    target_modules : list[str] or None
        Linear modules to adapt.  Defaults to all Mistral attention + MLP
        projections.
    use_dora : bool
        Enable Weight-Decomposed Low-Rank Adaptation (DoRA).
    use_gradient_checkpointing : bool
        Trade compute for memory with gradient checkpointing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    model
        The PEFT-wrapped model.
    """
    _require_unsloth()
    from unsloth import FastLanguageModel

    if target_modules is None:
        target_modules = DEFAULT_MISTRAL_TARGET_MODULES

    # Unsloth uses "unsloth" as a special gradient checkpointing mode that
    # is more memory-efficient than the standard HF implementation.
    gc_value = "unsloth" if use_gradient_checkpointing else False

    logger.info(
        "Applying Unsloth PEFT: r=%d, alpha=%d, dropout=%.3f, dora=%s, "
        "grad_ckpt=%s, targets=%s",
        r,
        lora_alpha,
        lora_dropout,
        use_dora,
        gc_value,
        target_modules,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        use_dora=use_dora,
        bias="none",
        use_gradient_checkpointing=gc_value,
        random_state=random_state,
    )
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_unsloth_model(model, tokenizer, save_path):
    """Save only the LoRA/DoRA adapter weights and the tokenizer.

    Parameters
    ----------
    model
        The PEFT-wrapped model.
    tokenizer
        The tokenizer returned by :func:`load_model_unsloth`.
    save_path : str or Path
        Directory to save into.
    """
    save_path = str(save_path)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    logger.info("Saving Unsloth adapter to %s", save_path)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def load_unsloth_model_for_inference(
    model_name,
    adapter_path,
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,
):
    """Reload the base model + trained adapter for inference.

    Parameters
    ----------
    model_name : str
        Same base model used during training.
    adapter_path : str
        Path to the saved adapter (output of :func:`save_unsloth_model`).
    max_seq_length : int
        Maximum sequence length.
    load_in_4bit : bool
        Whether to load the base model in 4-bit.
    dtype : torch.dtype or None
        Compute dtype.

    Returns
    -------
    model, tokenizer
    """
    _require_unsloth()
    from unsloth import FastLanguageModel
    from peft import PeftModel

    logger.info(
        "Reloading base model + adapter for inference: base=%s, adapter=%s",
        model_name,
        adapter_path,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
    )
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    FastLanguageModel.for_inference(model)
    return model, tokenizer
