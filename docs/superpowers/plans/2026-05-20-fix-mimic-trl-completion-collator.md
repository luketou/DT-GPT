# Fix MIMIC TRL Completion Collator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the MIMIC BioMistral DoRA training script run past data-collator and `SFTTrainer` construction in environments with TRL 0.24.0 while preserving completion-only loss masking.

**Architecture:** Replace the removed TRL `DataCollatorForCompletionOnlyLM` dependency with a repo-local completion-only collator that masks all prompt/template/padding labels to `-100` and trains only on target tokens. Build `SFTTrainer` kwargs by inspecting the installed TRL constructor so the script works with both older TRL APIs (`tokenizer`, `max_seq_length`, `packing`, `dataset_text_field`) and current TRL 0.24 APIs (`processing_class`, no legacy kwargs).

**Tech Stack:** Python 3.11, `torch`, `transformers`, `trl`, Hugging Face `datasets`, `unittest`.

---

## Root-Cause Evidence

- Log failure: `logs/mimic_dora_resume1395_to4185_37368.err` crashes at `pipeline/data_processors/DataProcessorBiomistral.py:305` with `ImportError: cannot import name 'DataCollatorForCompletionOnlyLM' from 'trl'`.
- Environment evidence: `conda run -n dtgpt-vllm python -m pip show trl` reports `Version: 0.24.0`.
- Installed package evidence: `grep -R "class DataCollatorForCompletionOnlyLM" .../site-packages/trl` finds no class; TRL 0.24 provides `trl.trainer.sft_trainer.DataCollatorForLanguageModeling`, which uses an explicit `completion_mask` rather than the removed template-scanning collator.
- Next compatibility risk: `trl.trainer.sft_trainer.SFTTrainer.__init__` in TRL 0.24 accepts `processing_class`, not the current script's `tokenizer`, `max_seq_length`, `packing`, and `dataset_text_field` kwargs.

## File Structure

- Modify: `pipeline/data_processors/DataProcessorBiomistral.py`
  - Add `CompletionOnlyDataCollator` near the top of the file.
  - Update `DataProcessorBiomistral.get_collator()` to use it for `collator_setting == "completion"`.
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
  - Add TRL-version-compatible `SFTTrainer` kwarg construction.
- Create: `tests/test_biomistral_completion_collator.py`
  - Unit tests for prompt masking, padding masking, and missing-template safety.

---

### Task 1: Add a failing regression test for completion-only masking

**Files:**
- Create: `tests/test_biomistral_completion_collator.py`
- Read: `pipeline/data_processors/DataProcessorBiomistral.py`

- [ ] **Step 1: Create the test file**

Create `tests/test_biomistral_completion_collator.py` with this exact content:

```python
import unittest

from pipeline.data_processors.DataProcessorBiomistral import CompletionOnlyDataCollator


class FakeTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, text, add_special_tokens=False):
        if text != "<patient_prediction>":
            raise AssertionError(f"unexpected tokenized text: {text!r}")
        return {"input_ids": [101, 102]}


class CompletionOnlyDataCollatorTest(unittest.TestCase):
    def test_masks_prompt_template_and_padding_labels(self):
        collator = CompletionOnlyDataCollator("<patient_prediction>", tokenizer=FakeTokenizer())

        batch = collator([
            {"input_ids": [10, 11, 101, 102, 20, 21]},
            {"input_ids": [101, 102, 30]},
        ])

        self.assertEqual(batch["input_ids"].tolist(), [
            [10, 11, 101, 102, 20, 21],
            [101, 102, 30, 0, 0, 0],
        ])
        self.assertEqual(batch["attention_mask"].tolist(), [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ])
        self.assertEqual(batch["labels"].tolist(), [
            [-100, -100, -100, -100, 20, 21],
            [-100, -100, 30, -100, -100, -100],
        ])

    def test_masks_everything_when_template_is_missing(self):
        collator = CompletionOnlyDataCollator("<patient_prediction>", tokenizer=FakeTokenizer())

        batch = collator([{"input_ids": [10, 11, 20]}])

        self.assertEqual(batch["labels"].tolist(), [[-100, -100, -100]])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails for the expected reason**

Run:

```bash
python -m unittest tests.test_biomistral_completion_collator -v
```

Expected result before implementation:

```text
ImportError: cannot import name 'CompletionOnlyDataCollator' from 'pipeline.data_processors.DataProcessorBiomistral'
```

---

### Task 2: Implement the repo-local completion-only collator

**Files:**
- Modify: `pipeline/data_processors/DataProcessorBiomistral.py:1-9`
- Modify: `pipeline/data_processors/DataProcessorBiomistral.py:11`
- Modify: `pipeline/data_processors/DataProcessorBiomistral.py:299-309`
- Test: `tests/test_biomistral_completion_collator.py`

- [ ] **Step 1: Add the `torch` import**

Change the import block at the top of `pipeline/data_processors/DataProcessorBiomistral.py` from:

```python
import __init__  # Do all imports
import logging
import wandb
import pandas as pd
import json
import numpy as np
from transformers import AutoTokenizer, LongT5Model, DataCollatorForSeq2Seq, T5Tokenizer, DataCollatorForLanguageModeling
from datasets import Dataset
import re
```

to:

```python
import __init__  # Do all imports
import logging
import wandb
import pandas as pd
import json
import numpy as np
import torch
from transformers import AutoTokenizer, LongT5Model, DataCollatorForSeq2Seq, T5Tokenizer, DataCollatorForLanguageModeling
from datasets import Dataset
import re
```

- [ ] **Step 2: Add `CompletionOnlyDataCollator` before `DataProcessorBiomistral`**

Insert this class immediately before `class DataProcessorBiomistral():`:

```python
class CompletionOnlyDataCollator:
    """Pad causal-LM batches and compute loss only after a response template.

    This replaces TRL's removed ``DataCollatorForCompletionOnlyLM`` for the
    repository's prompt format:

        <prompt text> <patient_prediction><target JSON>

    Labels before and including ``response_template`` are set to ``-100``.
    Padding labels are also set to ``-100``.
    """

    def __init__(self, response_template, tokenizer, label_pad_token_id=-100):
        self.response_template = response_template
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.response_token_ids = tokenizer(
            response_template,
            add_special_tokens=False,
        )["input_ids"]

        if not self.response_token_ids:
            raise ValueError("response_template tokenized to an empty token list")

    def __call__(self, features):
        input_ids = [list(feature["input_ids"]) for feature in features]
        attention_masks = [
            list(feature.get("attention_mask", [1] * len(feature["input_ids"])))
            for feature in features
        ]
        completion_masks = [self._build_completion_mask(ids) for ids in input_ids]

        max_length = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        padded_completion_masks = []

        for ids, attention_mask, completion_mask in zip(input_ids, attention_masks, completion_masks):
            pad_length = max_length - len(ids)
            if getattr(self.tokenizer, "padding_side", "right") == "left":
                padded_input_ids.append([self.tokenizer.pad_token_id] * pad_length + ids)
                padded_attention_masks.append([0] * pad_length + attention_mask)
                padded_completion_masks.append([0] * pad_length + completion_mask)
            else:
                padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_length)
                padded_attention_masks.append(attention_mask + [0] * pad_length)
                padded_completion_masks.append(completion_mask + [0] * pad_length)

        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
        }
        completion_mask_tensor = torch.tensor(padded_completion_masks, dtype=torch.bool)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = self.label_pad_token_id
        labels[~completion_mask_tensor] = self.label_pad_token_id
        batch["labels"] = labels
        return batch

    def _build_completion_mask(self, input_ids):
        response_start = self._find_subsequence(input_ids, self.response_token_ids)
        if response_start is None:
            logging.warning(
                "Response template %r was not found in tokenized sample; masking all labels.",
                self.response_template,
            )
            return [0] * len(input_ids)

        response_end = response_start + len(self.response_token_ids)
        return [0] * response_end + [1] * (len(input_ids) - response_end)

    @staticmethod
    def _find_subsequence(sequence, subsequence):
        max_start = len(sequence) - len(subsequence)
        for start in range(max_start + 1):
            if sequence[start:start + len(subsequence)] == subsequence:
                return start
        return None
```

- [ ] **Step 3: Update `get_collator()` to use the local collator**

Replace the existing completion branch:

```python
        elif self.collator_setting == "completion":
            from trl import DataCollatorForCompletionOnlyLM
            self.data_collator = DataCollatorForCompletionOnlyLM(self.response_template, tokenizer=self.tokenizer)
```

with:

```python
        elif self.collator_setting == "completion":
            self.data_collator = CompletionOnlyDataCollator(self.response_template, tokenizer=self.tokenizer)
```

- [ ] **Step 4: Run the focused collator test**

Run:

```bash
python -m unittest tests.test_biomistral_completion_collator -v
```

Expected result:

```text
test_masks_everything_when_template_is_missing ... ok
test_masks_prompt_template_and_padding_labels ... ok
```

---

### Task 3: Make `SFTTrainer` construction compatible with old and new TRL APIs

**Files:**
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py:498-512`

- [ ] **Step 1: Replace the current trainer kwarg block**

Replace:

```python
            sft_trainer_kwargs = {}
            if "dataset_num_proc" in inspect.signature(SFTTrainer).parameters:
                sft_trainer_kwargs["dataset_num_proc"] = sft_dataset_num_proc

            trainer = SFTTrainer(
                model=model,
                train_dataset=training_dataset,
                eval_dataset=validation_dataset,
                tokenizer=dp.tokenizer,
                data_collator=data_collator,
                max_seq_length=SEQUENCE_MAX_LENGTH_IN_TOKENS,
                args=train_params,
                packing=False,
                dataset_text_field="concatenated_text",
                **sft_trainer_kwargs,
            )
```

with:

```python
            sft_trainer_signature = inspect.signature(SFTTrainer).parameters
            sft_trainer_kwargs = {
                "model": model,
                "train_dataset": training_dataset,
                "eval_dataset": validation_dataset,
                "data_collator": data_collator,
                "args": train_params,
            }

            if "processing_class" in sft_trainer_signature:
                sft_trainer_kwargs["processing_class"] = dp.tokenizer
            elif "tokenizer" in sft_trainer_signature:
                sft_trainer_kwargs["tokenizer"] = dp.tokenizer

            if "max_seq_length" in sft_trainer_signature:
                sft_trainer_kwargs["max_seq_length"] = SEQUENCE_MAX_LENGTH_IN_TOKENS
            if "packing" in sft_trainer_signature:
                sft_trainer_kwargs["packing"] = False
            if "dataset_text_field" in sft_trainer_signature:
                sft_trainer_kwargs["dataset_text_field"] = "concatenated_text"
            if "dataset_num_proc" in sft_trainer_signature:
                sft_trainer_kwargs["dataset_num_proc"] = sft_dataset_num_proc

            trainer = SFTTrainer(**sft_trainer_kwargs)
```

- [ ] **Step 2: Verify the installed TRL signature is covered**

Run:

```bash
conda run -n dtgpt-vllm python -c "import inspect; from trl import SFTTrainer; print(inspect.signature(SFTTrainer))"
```

Expected result in the failing environment: output includes `processing_class` and does not include `tokenizer`, `max_seq_length`, `packing`, or `dataset_text_field`.

---

### Task 4: Run syntax and import validation

**Files:**
- Validate: `pipeline/data_processors/DataProcessorBiomistral.py`
- Validate: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
- Validate: `tests/test_biomistral_completion_collator.py`

- [ ] **Step 1: Compile changed Python files**

Run:

```bash
python -m py_compile \
  pipeline/data_processors/DataProcessorBiomistral.py \
  1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py \
  tests/test_biomistral_completion_collator.py
```

Expected result: command exits with status `0` and prints no traceback.

- [ ] **Step 2: Run the focused unit test in the training environment**

Run:

```bash
conda run -n dtgpt-vllm python -m unittest tests.test_biomistral_completion_collator -v
```

Expected result:

```text
test_masks_everything_when_template_is_missing ... ok
test_masks_prompt_template_and_padding_labels ... ok
```

If the exact test order differs, both tests must still report `ok`.

- [ ] **Step 3: Run the repo syntax smoke check**

Run:

```bash
conda run -n dtgpt-vllm python -m compileall pipeline 1_experiments
```

Expected result: command exits with status `0`. Existing unrelated warnings are acceptable; syntax errors are not.

---

### Task 5: Run a short non-destructive smoke check for collator and trainer construction

**Files:**
- Validate behavior of: `pipeline/data_processors/DataProcessorBiomistral.py`
- Validate behavior of: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`

- [ ] **Step 1: Verify the old failing import is no longer present**

Run:

```bash
rg -n "DataCollatorForCompletionOnlyLM" pipeline/data_processors/DataProcessorBiomistral.py
```

Expected result: no matches.

- [ ] **Step 2: Verify the local collator returns trainable labels only after `<patient_prediction>`**

Run:

```bash
conda run -n dtgpt-vllm python - <<'PY'
from pipeline.data_processors.DataProcessorBiomistral import CompletionOnlyDataCollator

class FakeTokenizer:
    pad_token_id = 0
    padding_side = "right"
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [101, 102]}

collator = CompletionOnlyDataCollator("<patient_prediction>", FakeTokenizer())
batch = collator([{"input_ids": [1, 101, 102, 2]}])
print(batch["labels"].tolist())
PY
```

Expected output:

```text
[[-100, -100, -100, 2]]
```

- [ ] **Step 3: Submit or rerun the original MIMIC job only after unit/syntax checks pass**

Run the same Slurm submission command that produced job `37368`, or directly rerun the script with the same CLI flags if interactive GPU resources are available.

Expected early-run evidence:

```text
Setting up data processor
Setting up model
Num params in model: 7285051392
Start training
```

The success criterion for this fix is reaching `Start training` without `ImportError: cannot import name 'DataCollatorForCompletionOnlyLM'` and without `SFTTrainer.__init__() got an unexpected keyword argument`.

---

## Self-Review

**Spec coverage:** The plan covers the observed `DataCollatorForCompletionOnlyLM` import failure, preserves completion-only label masking, and covers the next TRL 0.24 `SFTTrainer` constructor incompatibility found during root-cause investigation.

**Placeholder scan:** No unresolved placeholder language or undefined test/function references remain.

**Type consistency:** The test imports `CompletionOnlyDataCollator`, and Task 2 defines exactly that class name. The trainer kwarg variable is consistently named `sft_trainer_kwargs`.

## Commit Guidance

Use the repo's Lore commit protocol. Suggested commit message:

```text
Keep MIMIC completion training compatible with current TRL

Constraint: dtgpt-vllm currently has trl 0.24.0, which removed DataCollatorForCompletionOnlyLM and changed SFTTrainer kwargs.
Rejected: Pinning TRL from code | environment mutation is brittle and does not protect future runs with newer TRL.
Confidence: high
Scope-risk: narrow
Directive: Preserve completion-only label masking; do not replace with full causal-LM loss without an explicit experiment decision.
Tested: python -m unittest tests.test_biomistral_completion_collator -v; python -m py_compile pipeline/data_processors/DataProcessorBiomistral.py 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py tests/test_biomistral_completion_collator.py
Not-tested: Full GPU training job unless rerun reaches Start training.
```
