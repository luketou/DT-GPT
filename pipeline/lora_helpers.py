from pathlib import Path

from peft import LoraConfig, TaskType


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
):
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(target_modules),
    )
    config.target_modules = list(target_modules)
    return config


def build_lora_adapter_path(experiment_model_path):
    return str(Path(experiment_model_path) / "fine_tuned_lora_adapter")
