# Transformers Eval Strategy Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DT-GPT experiment scripts work with both legacy Transformers versions that accept `evaluation_strategy` and the current `dtgpt-vllm` Transformers 5.5.0 install that accepts `eval_strategy`.

**Architecture:** Add one reusable Hugging Face compatibility helper in `pipeline/` that inspects the installed `TrainingArguments.__init__` signature and normalizes only the renamed evaluation strategy keyword. Update the three experiment scripts that construct `TrainingArguments` to use the helper while keeping their existing training behavior unchanged.

**Tech Stack:** Python 3, Hugging Face Transformers/TRL, unittest, existing DT-GPT `pipeline` package.

---

## Root Cause Evidence

- Error log: `/home/r15543056/trajectory_forecast/DT-GPT/logs/mimic_dora_resume1395_to4185_37370.err`
- Failing file: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py:465-487`
- Hard failure: `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`
- Installed failing environment evidence: `/home/r15543056/miniconda3/envs/dtgpt-vllm/lib/python3.11/site-packages/transformers/__init__.py` declares `__version__ = "5.5.0"`.
- Installed API evidence: `/home/r15543056/miniconda3/envs/dtgpt-vllm/lib/python3.11/site-packages/transformers/training_args.py:1059` defines `eval_strategy`, not `evaluation_strategy`.
- Repo-wide call sites found with `rg -n "evaluation_strategy|eval_strategy|TrainingArguments" pipeline 1_experiments -g '*.py'`:
  - `1_experiments/2024_02_05_critical_vars/4_dt_gpt_instruction/2024_03_21_biomistral_td_bd/dt_gpt_fft_2024_03_21_biomistral_template_descr_bd.py:231-255`
  - `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py:465-487`
  - `1_experiments/2025_02_03_adni/3_dt_gpt/2025_02_03_dt_gpt_train_full.py:188-205`

## File Structure

- Create: `pipeline/hf_training_args.py`
  - Responsibility: centralize compatibility handling for `TrainingArguments` keyword renames.
  - Public functions:
    - `normalize_training_argument_kwargs(kwargs, training_arguments_cls) -> dict`
    - `create_training_arguments(**kwargs) -> transformers.TrainingArguments`
- Create: `tests/test_hf_training_args.py`
  - Responsibility: prove the helper maps `evaluation_strategy` to `eval_strategy` for modern APIs, preserves `evaluation_strategy` for legacy APIs, and rejects ambiguous duplicate strategy keys.
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
  - Replace direct `TrainingArguments` construction with `create_training_arguments`.
- Modify: `1_experiments/2024_02_05_critical_vars/4_dt_gpt_instruction/2024_03_21_biomistral_td_bd/dt_gpt_fft_2024_03_21_biomistral_template_descr_bd.py`
  - Same compatibility update for the analogous BioMistral experiment.
- Modify: `1_experiments/2025_02_03_adni/3_dt_gpt/2025_02_03_dt_gpt_train_full.py`
  - Same compatibility update for the ADNI training script.

---

### Task 1: Add failing compatibility tests

**Files:**
- Create: `tests/test_hf_training_args.py`
- Create later: `pipeline/hf_training_args.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hf_training_args.py` with this exact content:

```python
import unittest

from pipeline.hf_training_args import normalize_training_argument_kwargs


class ModernTrainingArguments:
    def __init__(self, output_dir=None, eval_strategy="no", save_strategy="no"):
        self.output_dir = output_dir
        self.eval_strategy = eval_strategy
        self.save_strategy = save_strategy


class LegacyTrainingArguments:
    def __init__(self, output_dir=None, evaluation_strategy="no", save_strategy="no"):
        self.output_dir = output_dir
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy


class TrainingArgumentsCompatibilityTest(unittest.TestCase):
    def test_maps_evaluation_strategy_to_eval_strategy_for_modern_transformers(self):
        normalized = normalize_training_argument_kwargs(
            {
                "output_dir": "model-output",
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
            },
            ModernTrainingArguments,
        )

        self.assertEqual(
            normalized,
            {
                "output_dir": "model-output",
                "eval_strategy": "steps",
                "save_strategy": "steps",
            },
        )

    def test_keeps_evaluation_strategy_for_legacy_transformers(self):
        normalized = normalize_training_argument_kwargs(
            {
                "output_dir": "model-output",
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
            },
            LegacyTrainingArguments,
        )

        self.assertEqual(
            normalized,
            {
                "output_dir": "model-output",
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
            },
        )

    def test_maps_eval_strategy_to_evaluation_strategy_for_legacy_transformers(self):
        normalized = normalize_training_argument_kwargs(
            {
                "output_dir": "model-output",
                "eval_strategy": "steps",
                "save_strategy": "steps",
            },
            LegacyTrainingArguments,
        )

        self.assertEqual(
            normalized,
            {
                "output_dir": "model-output",
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
            },
        )

    def test_rejects_duplicate_eval_strategy_aliases(self):
        with self.assertRaisesRegex(ValueError, "Both 'evaluation_strategy' and 'eval_strategy'"):
            normalize_training_argument_kwargs(
                {
                    "output_dir": "model-output",
                    "evaluation_strategy": "steps",
                    "eval_strategy": "epoch",
                },
                ModernTrainingArguments,
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_hf_training_args -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'pipeline.hf_training_args'`.

### Task 2: Implement the compatibility helper

**Files:**
- Create: `pipeline/hf_training_args.py`
- Test: `tests/test_hf_training_args.py`

- [ ] **Step 1: Add minimal helper implementation**

Create `pipeline/hf_training_args.py` with this exact content:

```python
"""Compatibility helpers for Hugging Face training argument API changes."""

import inspect


def normalize_training_argument_kwargs(kwargs, training_arguments_cls):
    """Return kwargs accepted by the installed TrainingArguments class.

    Transformers 5 uses ``eval_strategy`` while older versions used
    ``evaluation_strategy``. Experiment scripts keep the older name for
    readability/history; this helper maps that alias at the boundary.
    """
    normalized_kwargs = dict(kwargs)
    parameters = inspect.signature(training_arguments_cls.__init__).parameters
    has_evaluation_strategy = "evaluation_strategy" in parameters
    has_eval_strategy = "eval_strategy" in parameters

    if "evaluation_strategy" in normalized_kwargs and "eval_strategy" in normalized_kwargs:
        raise ValueError("Both 'evaluation_strategy' and 'eval_strategy' were provided.")

    if "evaluation_strategy" in normalized_kwargs and not has_evaluation_strategy and has_eval_strategy:
        normalized_kwargs["eval_strategy"] = normalized_kwargs.pop("evaluation_strategy")
    elif "eval_strategy" in normalized_kwargs and not has_eval_strategy and has_evaluation_strategy:
        normalized_kwargs["evaluation_strategy"] = normalized_kwargs.pop("eval_strategy")

    return normalized_kwargs


def create_training_arguments(**kwargs):
    """Create ``transformers.TrainingArguments`` with keyword compatibility."""
    from transformers import TrainingArguments

    return TrainingArguments(**normalize_training_argument_kwargs(kwargs, TrainingArguments))
```

- [ ] **Step 2: Run the focused helper tests**

Run:

```bash
python -m unittest tests.test_hf_training_args -v
```

Expected: PASS with all 4 tests reporting `ok`.

### Task 3: Update the failing MIMIC DoRA experiment

**Files:**
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`

- [ ] **Step 1: Replace the import**

Change:

```python
from transformers import TrainingArguments
```

to:

```python
from pipeline.hf_training_args import create_training_arguments
```

- [ ] **Step 2: Replace the constructor call**

Change:

```python
train_params = TrainingArguments(
```

to:

```python
train_params = create_training_arguments(
```

Keep every existing keyword argument unchanged, including `evaluation_strategy="steps"`.

- [ ] **Step 3: Verify the failing constructor no longer rejects the keyword**

Run:

```bash
/home/r15543056/miniconda3/envs/dtgpt-vllm/bin/python - <<'PY'
from pipeline.hf_training_args import normalize_training_argument_kwargs
from transformers import TrainingArguments
kwargs = normalize_training_argument_kwargs(
    {"output_dir": "/tmp/dtgpt-check", "evaluation_strategy": "steps"},
    TrainingArguments,
)
print(kwargs)
PY
```

Expected output includes:

```text
'eval_strategy': 'steps'
```

### Task 4: Update the other legacy TrainingArguments call sites

**Files:**
- Modify: `1_experiments/2024_02_05_critical_vars/4_dt_gpt_instruction/2024_03_21_biomistral_td_bd/dt_gpt_fft_2024_03_21_biomistral_template_descr_bd.py`
- Modify: `1_experiments/2025_02_03_adni/3_dt_gpt/2025_02_03_dt_gpt_train_full.py`

- [ ] **Step 1: Replace each direct import**

For each file, change:

```python
from transformers import TrainingArguments
```

to:

```python
from pipeline.hf_training_args import create_training_arguments
```

- [ ] **Step 2: Replace each constructor call**

For each file, change:

```python
train_params = TrainingArguments(
```

to:

```python
train_params = create_training_arguments(
```

Keep each existing `evaluation_strategy=...` expression unchanged.

- [ ] **Step 3: Confirm there are no remaining direct legacy calls**

Run:

```bash
rg -n "from transformers import TrainingArguments|TrainingArguments\(|evaluation_strategy|eval_strategy" pipeline 1_experiments -g '*.py'
```

Expected: only `evaluation_strategy` keyword values inside `create_training_arguments(...)` calls and helper/test references remain; no `from transformers import TrainingArguments` in the three modified scripts.

### Task 5: Validate syntax and focused behavior

**Files:**
- All modified files from Tasks 1-4.

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m unittest tests.test_hf_training_args -v
```

Expected: PASS with all 4 tests reporting `ok`.

- [ ] **Step 2: Run syntax check on touched code**

Run:

```bash
python -m py_compile pipeline/hf_training_args.py \
  1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py \
  1_experiments/2024_02_05_critical_vars/4_dt_gpt_instruction/2024_03_21_biomistral_td_bd/dt_gpt_fft_2024_03_21_biomistral_template_descr_bd.py \
  1_experiments/2025_02_03_adni/3_dt_gpt/2025_02_03_dt_gpt_train_full.py
```

Expected: command exits 0 with no output.

- [ ] **Step 3: Run repository lightweight syntax check when time allows**

Run:

```bash
python -m compileall pipeline 1_experiments
```

Expected: command exits 0. If this fails in unrelated legacy scripts, record the unrelated path and keep the focused `py_compile` result as the blocking evidence for this fix.

### Task 6: Commit the fix with Lore protocol

**Files:**
- `pipeline/hf_training_args.py`
- `tests/test_hf_training_args.py`
- `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
- `1_experiments/2024_02_05_critical_vars/4_dt_gpt_instruction/2024_03_21_biomistral_td_bd/dt_gpt_fft_2024_03_21_biomistral_template_descr_bd.py`
- `1_experiments/2025_02_03_adni/3_dt_gpt/2025_02_03_dt_gpt_train_full.py`

- [ ] **Step 1: Review diff**

Run:

```bash
git diff -- pipeline/hf_training_args.py tests/test_hf_training_args.py \
  1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py \
  1_experiments/2024_02_05_critical_vars/4_dt_gpt_instruction/2024_03_21_biomistral_td_bd/dt_gpt_fft_2024_03_21_biomistral_template_descr_bd.py \
  1_experiments/2025_02_03_adni/3_dt_gpt/2025_02_03_dt_gpt_train_full.py
```

Expected: diff only adds the compatibility helper/tests and swaps imports/constructor names.

- [ ] **Step 2: Commit with Lore protocol**

Run:

```bash
git add pipeline/hf_training_args.py tests/test_hf_training_args.py \
  1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py \
  1_experiments/2024_02_05_critical_vars/4_dt_gpt_instruction/2024_03_21_biomistral_td_bd/dt_gpt_fft_2024_03_21_biomistral_template_descr_bd.py \
  1_experiments/2025_02_03_adni/3_dt_gpt/2025_02_03_dt_gpt_train_full.py

git commit -m "Keep experiments runnable across Transformers strategy rename" \
  -m "Constraint: dtgpt-vllm currently has Transformers 5.5.0, whose TrainingArguments constructor accepts eval_strategy instead of evaluation_strategy.\nRejected: Rename all scripts directly to eval_strategy | would break older pinned environments that still accept evaluation_strategy.\nConfidence: high\nScope-risk: narrow\nDirective: Route future TrainingArguments compatibility changes through pipeline/hf_training_args.py instead of per-script conditionals.\nTested: python -m unittest tests.test_hf_training_args -v; python -m py_compile touched files\nNot-tested: full MIMIC training resume job, because it requires GPU cluster runtime and licensed data paths."
```

Expected: commit succeeds.

---

## Self-Review

- Spec coverage: The plan addresses the observed MIMIC failure and the two other repo call sites with the same legacy keyword.
- Placeholder scan: No `TBD`, `TODO`, or unspecified implementation steps remain.
- Type consistency: The helper names are consistent across tests and import replacements: `normalize_training_argument_kwargs` and `create_training_arguments`.
