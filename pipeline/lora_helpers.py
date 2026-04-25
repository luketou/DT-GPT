from pathlib import Path
import inspect

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM


DEFAULT_MISTRAL_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def build_mistral_lora_config(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=DEFAULT_MISTRAL_LORA_TARGET_MODULES,
    use_dora=False,
):
    config_kwargs = {
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
        "target_modules": list(target_modules),
    }
    if use_dora:
        if "use_dora" not in inspect.signature(LoraConfig).parameters:
            raise RuntimeError(
                "DoRA requires a PEFT version whose LoraConfig supports use_dora. "
                "Upgrade peft before running with --use-dora."
            )
        config_kwargs["use_dora"] = True

    config = LoraConfig(**config_kwargs)
    config.target_modules = list(target_modules)
    return config


def build_lora_adapter_path(experiment_model_path):
    return str(Path(experiment_model_path) / "fine_tuned_lora_adapter")


def apply_lora_to_model(model, lora_config, gradient_checkpointing=False):
    if gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_lora_model_for_inference(model_name_or_path, adapter_path, model_load_kwargs):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_load_kwargs,
    )
    return PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
    )
