# MIMIC BioMistral LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the current MIMIC BioMistral full fine-tuning path to PEFT LoRA so training no longer OOMs when AdamW initializes full-model optimizer state on V100 GPUs.

**Architecture:** Keep the existing data preparation, `SFTTrainer`, and evaluation flow, but insert a small PEFT helper layer that creates a Mistral-specific `LoraConfig`, wraps the loaded base model, saves only adapter weights, and reloads the base model plus adapter for evaluation. Expose LoRA hyperparameters from the launcher script and make the Slurm job use the LoRA path by default.

**Tech Stack:** Python 3.8, `transformers`, `trl`, `peft`, `torch`, `unittest`, Slurm

---

## File Structure

- Create: `pipeline/lora_helpers.py`
  Responsibility: centralize Mistral LoRA config creation, adapter wrapping, adapter path naming, and adapter reload helpers.
- Create: `tests/test_lora_helpers.py`
  Responsibility: regression coverage for LoRA config defaults and adapter path naming.
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
  Responsibility: switch the MIMIC BioMistral trainer from full fine-tuning to LoRA adapter training and adapter reload for evaluation.
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py`
  Responsibility: expose LoRA CLI flags and pass them into the experiment runner.
- Modify: `job/submit_mimic_train_v100.sh`
  Responsibility: make the batch job invoke the LoRA path explicitly and log the new training mode.

### Task 1: Add LoRA Helper Coverage

**Files:**
- Create: `tests/test_lora_helpers.py`
- Create: `pipeline/lora_helpers.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

from pipeline.lora_helpers import (
    DEFAULT_MISTRAL_LORA_TARGET_MODULES,
    build_lora_adapter_path,
    build_mistral_lora_config,
)


class LoraHelperTests(unittest.TestCase):
    def test_build_mistral_lora_config_uses_expected_defaults(self):
        config = build_mistral_lora_config()
        self.assertEqual(config.r, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertAlmostEqual(config.lora_dropout, 0.05)
        self.assertEqual(config.bias, "none")
        self.assertEqual(config.task_type.value, "CAUSAL_LM")
        self.assertEqual(
            tuple(config.target_modules),
            DEFAULT_MISTRAL_LORA_TARGET_MODULES,
        )

    def test_build_lora_adapter_path_uses_adapter_directory_name(self):
        self.assertEqual(
            build_lora_adapter_path("/tmp/experiment/"),
            "/tmp/experiment/fine_tuned_lora_adapter",
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n dtgpt python -m unittest discover -s tests -p 'test_lora_helpers.py'`

Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.lora_helpers'`

- [ ] **Step 3: Write minimal implementation**

```python
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
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(target_modules),
    )


def build_lora_adapter_path(experiment_model_path):
    return str(Path(experiment_model_path) / "fine_tuned_lora_adapter")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n dtgpt python -m unittest discover -s tests -p 'test_lora_helpers.py'`

Expected: PASS with `Ran 2 tests`

- [ ] **Step 5: Commit**

```bash
git add pipeline/lora_helpers.py tests/test_lora_helpers.py
git commit -m "feat: add LoRA helpers for BioMistral training"
```

### Task 2: Switch the MIMIC Trainer From Full FT to LoRA

**Files:**
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py:18-34`
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py:40-51`
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py:269-365`
- Modify: `pipeline/lora_helpers.py`
- Test: `tests/test_lora_helpers.py`

- [ ] **Step 1: Write the failing test**

Append this test to `tests/test_lora_helpers.py`:

```python
    def test_build_mistral_lora_config_supports_overrides(self):
        config = build_mistral_lora_config(r=8, lora_alpha=16, lora_dropout=0.1)
        self.assertEqual(config.r, 8)
        self.assertEqual(config.lora_alpha, 16)
        self.assertAlmostEqual(config.lora_dropout, 0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n dtgpt python -m unittest discover -s tests -p 'test_lora_helpers.py'`

Expected: FAIL because `build_mistral_lora_config()` does not yet expose or preserve the override behavior used by the training entry point.

- [ ] **Step 3: Write minimal implementation**

Update `pipeline/lora_helpers.py` to include PEFT application and reload helpers:

```python
from pathlib import Path

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
):
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(target_modules),
    )


def apply_lora_to_model(model, lora_config, gradient_checkpointing=False):
    if gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def build_lora_adapter_path(experiment_model_path):
    return str(Path(experiment_model_path) / "fine_tuned_lora_adapter")


def load_lora_model_for_inference(
    model_name_or_path,
    adapter_path,
    model_load_kwargs,
):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_load_kwargs,
    )
    return PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
    )
```

Update the trainer in `dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`:

```python
from pipeline.lora_helpers import (
    apply_lora_to_model,
    build_lora_adapter_path,
    build_mistral_lora_config,
    load_lora_model_for_inference,
)
```

```python
    def run(
        self,
        debug=False,
        verbose=True,
        wandb_prefix_name="DT-GPT - Meditron FFT - Completion Loss - Forecast: ",
        wandb_group_name="DT-GPT - Meditron FFT - Completion",
        train_set="TRAIN",
        validation_set="VALIDATION",
        test_set="TEST",
        learning_rate=1e-5,
        batch_size_training=1,
        batch_size_validation=1,
        weight_decay=0.1,
        gradient_accumulation=1,
        num_train_epochs=1.0,
        eval_interval=0.25,
        warmup_ratio=0.1,
        lr_scheduler="cosine",
        gradient_checkpointing=False,
        logging_steps=10,
        nr_days_forecasting=91,
        seq_max_len_in_tokens=4000,
        decimal_precision=1,
        gen_num_beams=1,
        gen_do_sample=False,
        eval_model_path=None,
        num_samples_to_generate=10,
        sample_merging_strategy="mean",
        max_new_tokens_to_generate=1200,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ):
```

```python
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_HF_NAME,
                **model_load_kwargs,
            )

            if gradient_checkpointing:
                model.config.use_cache = False

            if use_lora:
                lora_config = build_mistral_lora_config(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                model = apply_lora_to_model(
                    model,
                    lora_config,
                    gradient_checkpointing=gradient_checkpointing,
                )
```

```python
            finetune_model_path = build_lora_adapter_path(experiment.model_path)
            model.save_pretrained(finetune_model_path)
```

```python
            model = load_lora_model_for_inference(
                MODEL_HF_NAME,
                finetune_model_path,
                get_model_load_kwargs(
                    experiment.model_cache_path,
                    training=False,
                ),
            )
```

```python
            model = load_lora_model_for_inference(
                MODEL_HF_NAME,
                eval_model_path,
                get_model_load_kwargs(
                    experiment.model_cache_path,
                    training=False,
                ),
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n dtgpt python -m unittest discover -s tests -p 'test_lora_helpers.py'`

Expected: PASS with `Ran 3 tests`

- [ ] **Step 5: Run syntax verification**

Run: `conda run -n dtgpt python -m compileall pipeline/lora_helpers.py 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`

Expected: `Compiling ...` lines and exit code `0`

- [ ] **Step 6: Commit**

```bash
git add pipeline/lora_helpers.py tests/test_lora_helpers.py 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py
git commit -m "feat: switch MIMIC BioMistral training to LoRA"
```

### Task 3: Expose LoRA Controls in the Launcher and Slurm Job

**Files:**
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py:20-70`
- Modify: `job/submit_mimic_train_v100.sh:1-70`

- [ ] **Step 1: Write the failing test**

Use a command-line smoke check as the failing test because this repository does not already have an import-friendly module for this dated launcher file.

Run:

```bash
conda run -n dtgpt python 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py --help
```

Expected: FAIL to show `--use-lora`, `--lora-r`, `--lora-alpha`, and `--lora-dropout` because those flags do not exist yet.

- [ ] **Step 2: Write minimal implementation**

Update the launcher parser:

```python
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
```

Pass the values into `experiment.run(...)`:

```python
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
```

Update the Slurm job so LoRA is used by default:

```bash
echo "Training mode: LoRA"
"${PYTHON_BIN}" 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py \
    --use-lora \
    --lora-r "${DTGPT_LORA_R:-16}" \
    --lora-alpha "${DTGPT_LORA_ALPHA:-32}" \
    --lora-dropout "${DTGPT_LORA_DROPOUT:-0.05}" \
    --gradient-checkpointing \
    --train-batch-size 1 \
    --validation-batch-size "${VALIDATION_BATCH_SIZE}" \
    --seq-max-len "${SEQ_MAX_LEN}" \
    --num-train-epochs 5 \
    --num-samples-to-generate "${NUM_SAMPLES_TO_GENERATE}" \
    --max-new-tokens-to-generate "${MAX_NEW_TOKENS}"
```

- [ ] **Step 3: Run the launcher check to verify it passes**

Run:

```bash
conda run -n dtgpt python 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py --help
```

Expected: PASS and the help text includes `--use-lora`, `--lora-r`, `--lora-alpha`, and `--lora-dropout`

- [ ] **Step 4: Run focused verification**

Run:

```bash
conda run -n dtgpt python -m compileall 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py
conda run -n dtgpt python -m unittest discover -s tests -p 'test_lora_helpers.py'
```

Expected: both commands exit `0`

- [ ] **Step 5: Commit**

```bash
git add 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py job/submit_mimic_train_v100.sh
git commit -m "feat: enable LoRA from the MIMIC training launcher"
```

### Task 4: Validate the End-to-End LoRA Training Path

**Files:**
- Modify: `job/submit_mimic_train_v100.sh` if verification reveals a safe default needs adjustment
- Test: `job/submit_mimic_train_v100.sh`

- [ ] **Step 1: Run a one-epoch smoke training**

Run:

```bash
DTGPT_SEQ_MAX_LEN=2048 \
DTGPT_LORA_R=16 \
DTGPT_LORA_ALPHA=32 \
DTGPT_LORA_DROPOUT=0.05 \
bash job/submit_mimic_train_v100.sh
```

Expected: the run gets past the first optimizer step without `torch.cuda.OutOfMemoryError` from `adamw.py`

- [ ] **Step 2: If the smoke run still exceeds memory, apply the smallest safe config reduction**

Update the job defaults only if Step 1 still fails:

```bash
SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-1536}"
```

Expected: keep the LoRA code unchanged and reduce only the launch-time sequence length.

- [ ] **Step 3: Re-run the smoke training**

Run:

```bash
DTGPT_SEQ_MAX_LEN=1536 bash job/submit_mimic_train_v100.sh
```

Expected: training starts, reaches optimizer initialization, and continues without OOM

- [ ] **Step 4: Commit**

```bash
git add job/submit_mimic_train_v100.sh
git commit -m "chore: tune LoRA MIMIC training defaults for V100"
```

## Self-Review

- Spec coverage: the plan covers the helper layer, trainer wiring, launcher flags, Slurm defaults, and runtime verification required to move from full fine-tuning to LoRA.
- Placeholder scan: no `TODO`, `TBD`, or unresolved file references remain.
- Type consistency: the same `use_lora`, `lora_r`, `lora_alpha`, and `lora_dropout` names are used in the helper, launcher, trainer, and job script.

