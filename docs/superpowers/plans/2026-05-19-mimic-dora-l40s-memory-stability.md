# MIMIC DoRA L40S Memory Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume the MIMIC DoRA run on two L40S GPUs without repeatedly dying during DF-to-string conversion, and make DF conversion robust enough that future runs do not depend on GPU count to survive preprocessing.

**Architecture:** Short term, modify the existing resume submit script in place so the already-reserved two L40S GPUs on node 201 are actually used by disabling Unsloth and enabling the existing distributed DeepSpeed path, while also requesting explicit CPU RAM because the observed failure occurs before training. Root fix, replace eager joblib conversion that materializes all converted samples at once with streaming conversion into a Hugging Face `Dataset`, add memory/progress instrumentation, and release raw DataFrame objects before tokenization/model load.

**Tech Stack:** Bash/Slurm, Python 3, pandas, joblib, Hugging Face `datasets`, TRL `SFTTrainer`, Transformers/PEFT/DeepSpeed, pytest, `python -m compileall`.

---

## Evidence and diagnosis

Observed files:
- `logs/mimic_dora_resume1395_to2100_37274.out`
- `logs/mimic_dora_resume1395_to2100_37274.err`
- `job/submit_mimic_dora_resume1395_to4185.sh`
- `job/submit_mimic_dora.sh`
- `pipeline/DFConversionHelpers.py`
- `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
- `pipeline/data_processors/DataProcessorBiomistral.py`

Important observations:
- `job/submit_mimic_dora_resume1395_to4185.sh` has been manually changed by the user to reserve two L40S GPUs, effectively taking all of node 201 for this run, but it still exports `DTGPT_USE_DISTRIBUTED=0`, so the Python run uses only one process/GPU unless the environment flags are changed.
- `job/submit_mimic_dora.sh` blocks `DTGPT_USE_UNSLOTH=1` with `DTGPT_USE_DISTRIBUTED=1`, so true two-GPU training must use standard PEFT/DoRA with Unsloth disabled, or the script exits intentionally.
- The failing log reaches `Converting DFs to Strings with joblib workers: 1 for 44667 tuples`, then the shell reports `Killed`. This happens before `Setting up model`, so adding GPUs alone cannot fix the immediate death. The likely failure mode is CPU RAM pressure or a long single-worker preprocessing phase being killed by the scheduler/cgroup.
- Root memory pressure sources in the current code:
  - full train/validation/test DataFrame lists stay in memory;
  - `training_events` becomes another list of tuples containing DataFrame copies;
  - `process_all_tuples_multiprocessing()` calls `Parallel(...)(...)`, materializing all conversion results before unpacking;
  - `DataProcessorBiomistral.preprocess_dataset()` then builds more full Python string lists and a Hugging Face dataset.

## File structure

- Modify: `job/submit_mimic_dora_resume1395_to4185.sh`
  - Existing resume entrypoint; keep this filename and change its defaults so the reserved two L40S GPUs on node 201 are actually used.
- Modify: `pipeline/DFConversionHelpers.py`
  - Add memory logging, safe sequential streaming conversion, and generator helpers that do not create a giant joblib result list when `n_jobs=1`.
- Modify: `pipeline/data_processors/DataProcessorBiomistral.py`
  - Add a dataset-building method that accepts converted-text records/generators and tokenizes without requiring separate full input/target Python lists.
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
  - Use streaming conversion for train/validation, delete raw DataFrame containers once no longer needed, and log memory at boundaries.
- Create: `tests/test_df_conversion_helpers.py`
  - Unit tests for streaming order, error reporting, and no joblib use for `n_jobs=1`.
- Create: `tests/test_data_processor_streaming.py`
  - Unit tests for constructing/tokenizing a minimal dataset from converted records without preserving redundant string columns.

---

### Task 1: Modify the existing resume script to use the two reserved L40S GPUs

**Files:**
- Modify: `job/submit_mimic_dora_resume1395_to4185.sh`

- [ ] **Step 1: Replace the existing Slurm/resource and distributed defaults in place**

Edit `job/submit_mimic_dora_resume1395_to4185.sh`. Keep the filename unchanged. The resulting file must contain this exact content unless local scheduler policy requires a different `--nodelist` spelling:

```bash
#!/bin/bash
#SBATCH --job-name="dtgpt-mimic-dora-1395to4185"
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --nodelist=node201
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=7-0:0
#SBATCH --output=logs/mimic_dora_resume1395_to4185_%j.out
#SBATCH --error=logs/mimic_dora_resume1395_to4185_%j.err
#SBATCH --chdir=/share/home/r15543056/trajectory_forecast/DT-GPT

set -euo pipefail

export DTGPT_RESUME_FROM_CHECKPOINT="${DTGPT_RESUME_FROM_CHECKPOINT:-/share/home/r15543056/trajectory_forecast/DT-GPT/3_results/raw_experiments/DT-GPTsetup/setup/2026_05_05___16_46_44_938674/models/checkpoint-1395}"
export DTGPT_MAX_STEPS="${DTGPT_MAX_STEPS:-4185}"
export DTGPT_CONDA_ENV="${DTGPT_CONDA_ENV:-dtgpt-vllm}"
export DTGPT_PATIENT_SPLIT_FRACTION="${DTGPT_PATIENT_SPLIT_FRACTION:-0.5}"
export DTGPT_SWEEP_CONFIGS="${DTGPT_SWEEP_CONFIGS:-16,32,32,3,8e-6}"
export DTGPT_LORA_DROPOUT="${DTGPT_LORA_DROPOUT:-0}"
export DTGPT_SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-4096}"
export DTGPT_NUM_SAMPLES_TO_GENERATE="${DTGPT_NUM_SAMPLES_TO_GENERATE:-1}"
export DTGPT_MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-128}"
export DTGPT_TRAIN_BATCH_SIZE="${DTGPT_TRAIN_BATCH_SIZE:-1}"
export DTGPT_VALIDATION_BATCH_SIZE="${DTGPT_VALIDATION_BATCH_SIZE:-1}"
export DTGPT_GRADIENT_CHECKPOINTING="${DTGPT_GRADIENT_CHECKPOINTING:-1}"
export DTGPT_USE_DORA="${DTGPT_USE_DORA:-1}"
export DTGPT_USE_UNSLOTH="${DTGPT_USE_UNSLOTH:-0}"
export DTGPT_USE_DISTRIBUTED="${DTGPT_USE_DISTRIBUTED:-1}"
export DTGPT_USE_DEEPSPEED="${DTGPT_USE_DEEPSPEED:-1}"
export DTGPT_NPROC_PER_NODE="${DTGPT_NPROC_PER_NODE:-2}"
export DTGPT_DEEPSPEED_CONFIG="${DTGPT_DEEPSPEED_CONFIG:-job/deepspeed_zero3_config.json}"
export DTGPT_RUN_SPLIT_SMOKE_CHECK="${DTGPT_RUN_SPLIT_SMOKE_CHECK:-1}"
export DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK="${DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK:-1}"
export DTGPT_DF_CONVERSION_N_JOBS="${DTGPT_DF_CONVERSION_N_JOBS:-1}"
export DTGPT_SFT_DATASET_NUM_PROC="${DTGPT_SFT_DATASET_NUM_PROC:-1}"

if [ ! -d "${DTGPT_RESUME_FROM_CHECKPOINT}" ]; then
    echo "Resume checkpoint not found: ${DTGPT_RESUME_FROM_CHECKPOINT}" >&2
    exit 1
fi

if [ ! -f "${DTGPT_DEEPSPEED_CONFIG}" ]; then
    echo "DeepSpeed config not found: ${DTGPT_DEEPSPEED_CONFIG}" >&2
    exit 1
fi

echo "Resume checkpoint: ${DTGPT_RESUME_FROM_CHECKPOINT}"
echo "Conda env: ${DTGPT_CONDA_ENV}"
echo "Target max global step: ${DTGPT_MAX_STEPS}"
echo "Reserved node: node201"
echo "GPUs requested: 2 L40S; distributed processes: ${DTGPT_NPROC_PER_NODE}"
echo "Training mode: standard PEFT DoRA + DeepSpeed; Unsloth disabled because this repo blocks Unsloth distributed training"
echo "Requested CPU memory: 160G because current failure occurs during CPU DF-to-string conversion before model setup"

bash job/submit_mimic_dora.sh
```

- [ ] **Step 2: Verify Bash syntax**

Run:

```bash
bash -n job/submit_mimic_dora_resume1395_to4185.sh
```

Expected: command exits with code `0` and prints nothing.

- [ ] **Step 3: Verify the existing script uses the intended resource and training flags**

Run:

```bash
grep -nE "nodelist|gres=gpu:2|--mem=160G|DTGPT_USE_UNSLOTH|DTGPT_USE_DISTRIBUTED|DTGPT_USE_DEEPSPEED|DTGPT_NPROC_PER_NODE" job/submit_mimic_dora_resume1395_to4185.sh
```

Expected output contains these lines or equivalent line numbers:

```text
#SBATCH --nodelist=node201
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
export DTGPT_USE_UNSLOTH="${DTGPT_USE_UNSLOTH:-0}"
export DTGPT_USE_DISTRIBUTED="${DTGPT_USE_DISTRIBUTED:-1}"
export DTGPT_USE_DEEPSPEED="${DTGPT_USE_DEEPSPEED:-1}"
export DTGPT_NPROC_PER_NODE="${DTGPT_NPROC_PER_NODE:-2}"
```

- [ ] **Step 4: Commit the in-place submit-script change**

Run:

```bash
git add job/submit_mimic_dora_resume1395_to4185.sh
git commit -m "Use reserved dual L40S capacity for MIMIC resume

Constraint: The user reserved node 201 with two L40S GPUs and wants the existing resume script modified in place.
Rejected: Creating a second resume script | it would make the active submit path ambiguous.
Rejected: Only requesting gpu:2 | the previous script requested two GPUs but disabled distributed execution.
Confidence: high
Scope-risk: narrow
Directive: Keep Unsloth disabled for distributed runs unless the repo gains explicit Unsloth multi-GPU support.
Tested: bash -n job/submit_mimic_dora_resume1395_to4185.sh; grep resource and distributed flags
Not-tested: sbatch execution on node 201"
```

---

### Task 2: Add streaming conversion and memory instrumentation

**Files:**
- Modify: `pipeline/DFConversionHelpers.py`
- Create: `tests/test_df_conversion_helpers.py`

- [ ] **Step 1: Write tests for streaming conversion**

Create `tests/test_df_conversion_helpers.py` with this exact content:

```python
import logging

import pytest

from pipeline.DFConversionHelpers import (
    iter_converted_tuples,
    log_memory_usage,
    process_all_tuples_multiprocessing,
    resolve_df_conversion_n_jobs,
)


def test_resolve_df_conversion_n_jobs_rejects_zero():
    with pytest.raises(ValueError, match="positive integer"):
        resolve_df_conversion_n_jobs(0)


def test_iter_converted_tuples_preserves_order_and_metadata(caplog):
    def convert(value):
        return f"input-{value}", f"target-{value}", {"idx": value}

    caplog.set_level(logging.INFO)

    records = list(iter_converted_tuples([(1,), (2,), (3,)], convert, log_every=2))

    assert records == [
        {"input_text": "input-1", "target_text": "target-1", "meta_data": {"idx": 1}},
        {"input_text": "input-2", "target_text": "target-2", "meta_data": {"idx": 2}},
        {"input_text": "input-3", "target_text": "target-3", "meta_data": {"idx": 3}},
    ]
    assert "Converting DFs to Strings: 2 / 3" in caplog.text


def test_process_all_tuples_multiprocessing_uses_streaming_path_for_one_worker(monkeypatch):
    def fail_if_joblib_parallel_is_called(*args, **kwargs):
        raise AssertionError("joblib Parallel must not be used for n_jobs=1")

    def convert(value):
        return f"input-{value}", f"target-{value}", {"idx": value}

    monkeypatch.setattr("pipeline.DFConversionHelpers.Parallel", fail_if_joblib_parallel_is_called)

    input_strings, target_strings, meta_data = process_all_tuples_multiprocessing(
        [(1,), (2,)],
        convert,
        n_jobs=1,
    )

    assert input_strings == ["input-1", "input-2"]
    assert target_strings == ["target-1", "target-2"]
    assert meta_data == [{"idx": 1}, {"idx": 2}]


def test_log_memory_usage_does_not_fail_without_psutil(caplog):
    caplog.set_level(logging.INFO)
    log_memory_usage("unit-test")
    assert "unit-test" in caplog.text
```

- [ ] **Step 2: Run tests and verify they fail before implementation**

Run:

```bash
pytest tests/test_df_conversion_helpers.py -v
```

Expected before implementation: collection fails with `ImportError` or tests fail because `iter_converted_tuples` and `log_memory_usage` are not defined.

- [ ] **Step 3: Implement streaming helpers**

Replace `pipeline/DFConversionHelpers.py` with this exact content:

```python
import logging
import os
import resource
import time

from joblib import Parallel, delayed
from tqdm import tqdm


def resolve_df_conversion_n_jobs(n_jobs=None):
    try:
        if n_jobs is not None:
            resolved_n_jobs = int(n_jobs)
        else:
            resolved_n_jobs = int(os.environ.get("DTGPT_DF_CONVERSION_N_JOBS", "1"))
    except ValueError as error:
        raise ValueError("DTGPT_DF_CONVERSION_N_JOBS must be a positive integer.") from error

    if resolved_n_jobs < 1:
        raise ValueError("DTGPT_DF_CONVERSION_N_JOBS must be a positive integer.")

    return resolved_n_jobs


def _rss_megabytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 10_000_000:
        return rss / (1024 * 1024)
    return rss / 1024


def log_memory_usage(label):
    logging.info("Memory usage at %s: maxrss=%.1f MB", label, _rss_megabytes())


def iter_converted_tuples(list_of_data_tuples, conversion_function, log_every=100):
    total = len(list_of_data_tuples)
    started_at = time.monotonic()
    log_memory_usage("df-conversion-start")

    for idx, data in enumerate(list_of_data_tuples, start=1):
        if idx == 1 or idx % log_every == 0 or idx == total:
            elapsed = time.monotonic() - started_at
            logging.info(
                "Converting DFs to Strings: %s / %s elapsed=%.1fs",
                idx,
                total,
                elapsed,
            )
            log_memory_usage(f"df-conversion-{idx}-of-{total}")

        string_input, string_output, meta_data = conversion_function(*data)
        yield {
            "input_text": string_input,
            "target_text": string_output,
            "meta_data": meta_data,
        }

    elapsed = time.monotonic() - started_at
    logging.info("Finished converting %s DFs to strings in %.1fs", total, elapsed)
    log_memory_usage("df-conversion-finished")


def process_all_tuples(list_of_data_tuples, conversion_function):
    list_input_strings = []
    list_target_strings = []
    list_meta_data = []

    for record in iter_converted_tuples(list_of_data_tuples, conversion_function, log_every=10):
        list_input_strings.append(record["input_text"])
        list_target_strings.append(record["target_text"])
        list_meta_data.append(record["meta_data"])

    return list_input_strings, list_target_strings, list_meta_data


def process_all_tuples_multiprocessing(list_of_data_tuples, conversion_function, n_jobs=None):
    resolved_n_jobs = resolve_df_conversion_n_jobs(n_jobs)
    logging.info(
        "Converting DFs to Strings with joblib workers: %s for %s tuples",
        resolved_n_jobs,
        len(list_of_data_tuples),
    )

    if resolved_n_jobs == 1:
        return process_all_tuples(list_of_data_tuples, conversion_function)

    results = Parallel(n_jobs=resolved_n_jobs)(
        delayed(conversion_function)(*data) for data in tqdm(list_of_data_tuples)
    )

    list_input_strings, list_target_strings, list_meta_data = zip(*results)
    return list(list_input_strings), list(list_target_strings), list(list_meta_data)
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
pytest tests/test_df_conversion_helpers.py -v
```

Expected: all four tests pass.

- [ ] **Step 5: Commit streaming helper changes**

Run:

```bash
git add pipeline/DFConversionHelpers.py tests/test_df_conversion_helpers.py
git commit -m "Make DF conversion observable and streaming for one worker

Constraint: The failed run was killed during CPU-side DF-to-string conversion before model setup.
Rejected: Increasing joblib workers | more workers copy large pandas objects and can increase RAM pressure.
Confidence: high
Scope-risk: moderate
Directive: Keep n_jobs=1 as the safe default for full MIMIC conversion unless memory accounting proves higher parallelism is safe.
Tested: pytest tests/test_df_conversion_helpers.py -v
Not-tested: Full 44,667 tuple conversion on Slurm"
```

---

### Task 3: Add generator-based dataset preprocessing

**Files:**
- Modify: `pipeline/data_processors/DataProcessorBiomistral.py`
- Create: `tests/test_data_processor_streaming.py`

- [ ] **Step 1: Write tests for preprocessing converted records**

Create `tests/test_data_processor_streaming.py` with this exact content:

```python
from pipeline.data_processors.DataProcessorBiomistral import DataProcessorBiomistral


class FakeTokenizer:
    eos_token = "<eos>"
    add_eos_token = True
    chat_template = ""

    def __call__(self, text, max_length, truncation):
        return {
            "input_ids": [[min(len(item), max_length)] for item in text],
            "attention_mask": [[1] for _ in text],
        }


def make_processor(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text("{}")
    processor = DataProcessorBiomistral.__new__(DataProcessorBiomistral)
    processor.tokenizer = FakeTokenizer()
    processor.max_total_length = 16
    processor.collator_setting = "completion"
    processor.response_template = "<patient_prediction>"
    return processor


def test_preprocess_converted_records_removes_redundant_text_columns(tmp_path):
    processor = make_processor(tmp_path)
    records = [
        {"input_text": "history A", "target_text": "target A"},
        {"input_text": "history B", "target_text": "target B"},
    ]

    dataset = processor.preprocess_converted_records(records, tokenize=True, keep_text_columns=False)

    assert len(dataset) == 2
    assert dataset.column_names == ["input_ids", "attention_mask", "concatenated_text"]
    assert dataset[0]["concatenated_text"] == "history A <patient_prediction> target A"
    assert dataset[0]["input_ids"] == [16]


def test_preprocess_converted_records_can_keep_text_columns(tmp_path):
    processor = make_processor(tmp_path)
    records = [
        {"input_text": "history A", "target_text": "target A"},
    ]

    dataset = processor.preprocess_converted_records(records, tokenize=False, keep_text_columns=True)

    assert dataset.column_names == ["input_text", "target_text", "concatenated_text"]
    assert dataset[0]["input_text"] == "history A <patient_prediction>"
    assert dataset[0]["target_text"] == "target A"
    assert dataset[0]["concatenated_text"] == "history A <patient_prediction> target A"
```

- [ ] **Step 2: Run tests and verify they fail before implementation**

Run:

```bash
pytest tests/test_data_processor_streaming.py -v
```

Expected before implementation: tests fail with `AttributeError: 'DataProcessorBiomistral' object has no attribute 'preprocess_converted_records'`.

- [ ] **Step 3: Add `preprocess_converted_records`**

Modify `pipeline/data_processors/DataProcessorBiomistral.py` inside class `DataProcessorBiomistral`, directly after `preprocess_dataset()`:

```python
    def preprocess_converted_records(self, converted_records, tokenize=True, keep_text_columns=False):
        logging.info("DataProcessor: building dataset from converted records")
        curr_dataset = Dataset.from_list(list(converted_records))

        if "meta_data" in curr_dataset.column_names:
            curr_dataset = curr_dataset.remove_columns(["meta_data"])

        def preprocess_function(samples):
            inputs = self.preprocess_inputs(samples["input_text"])
            targets = self.preprocess_outputs(samples["target_text"])
            concat_text = [str(input_text) + " " + str(target_text) for input_text, target_text in zip(inputs, targets)]

            if tokenize:
                model_inputs = self.tokenizer(
                    text=concat_text,
                    max_length=self.max_total_length,
                    truncation=True,
                )
            else:
                model_inputs = {}

            if keep_text_columns:
                model_inputs["input_text"] = inputs
                model_inputs["target_text"] = targets

            model_inputs["concatenated_text"] = concat_text
            return model_inputs

        remove_columns = [] if keep_text_columns else [
            column_name for column_name in ["input_text", "target_text"] if column_name in curr_dataset.column_names
        ]
        tokenized_dataset = curr_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=remove_columns,
        )
        logging.info("DataProcessor: finished building dataset from converted records")
        return tokenized_dataset
```

- [ ] **Step 4: Run data processor tests**

Run:

```bash
pytest tests/test_data_processor_streaming.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Run combined unit tests**

Run:

```bash
pytest tests/test_df_conversion_helpers.py tests/test_data_processor_streaming.py -v
```

Expected: all six tests pass.

- [ ] **Step 6: Commit data processor changes**

Run:

```bash
git add pipeline/data_processors/DataProcessorBiomistral.py tests/test_data_processor_streaming.py
git commit -m "Build training datasets from converted records

Constraint: Full MIMIC string lists create avoidable duplicate memory before tokenization.
Rejected: Rewriting the whole data processor | a focused converted-records entrypoint preserves existing callers.
Confidence: medium
Scope-risk: moderate
Directive: Do not remove preprocess_dataset until all older experiment scripts have been migrated or verified.
Tested: pytest tests/test_df_conversion_helpers.py tests/test_data_processor_streaming.py -v
Not-tested: Full TRL trainer construction with the streamed dataset"
```

---

### Task 4: Wire streaming conversion into the MIMIC DoRA experiment and free raw objects

**Files:**
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`

- [ ] **Step 1: Update imports**

In `dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`, replace this import:

```python
from pipeline.DFConversionHelpers import process_all_tuples_multiprocessing
```

with this import:

```python
from pipeline.DFConversionHelpers import iter_converted_tuples, log_memory_usage, process_all_tuples_multiprocessing
```

- [ ] **Step 2: Add memory logs after loading raw data**

Directly after line that loads `test_full_constants, test_full_events`, add:

```python
        log_memory_usage("after-loading-train-validation-test-dfs")
```

- [ ] **Step 3: Release unused path lists and full test constants after splitting**

Directly after the `logging.info("Using eval backend %s; ...")` block that logs test shard information, add:

```python
        del training_full_paths, validation_full_paths, test_full_paths
        del test_full_constants
        gc.collect()
        log_memory_usage("after-splitting-and-dropping-unused-paths")
```

- [ ] **Step 4: Replace eager training conversion with streaming dataset construction**

Replace the current block from:

```python
            # apply multiprocessing based DF to string conversion to speed up process
            training_input_strings, training_target_strings, training_meta_data = process_all_tuples_multiprocessing(
                training_events,
                conversion_function,
                n_jobs=df_conversion_n_jobs,
            )

            # Print one example
            logging.info("Example of input: " + training_input_strings[0])
            logging.info("Example of target: " + training_target_strings[0])

            logging.info("Example of input: " + training_input_strings[1])
            logging.info("Example of target: " + training_target_strings[1])
```

with:

```python
            training_records = list(iter_converted_tuples(training_events, conversion_function, log_every=100))
            if len(training_records) >= 2:
                logging.info("Example of input: " + training_records[0]["input_text"])
                logging.info("Example of target: " + training_records[0]["target_text"])
                logging.info("Example of input: " + training_records[1]["input_text"])
                logging.info("Example of target: " + training_records[1]["target_text"])
            del training_events
            del training_full_constants, training_full_events
            gc.collect()
            log_memory_usage("after-training-string-conversion-and-raw-train-free")
```

This still materializes converted records once, but avoids joblib’s giant result list and frees raw train DataFrames before tokenization/model setup. Task 5 below can convert this to disk-backed Arrow if memory is still too high.

- [ ] **Step 5: Replace eager validation conversion with streaming records**

Replace the current validation conversion call:

```python
            validation_input_strings, validation_target_strings, validation_meta_data = process_all_tuples_multiprocessing(
                validation_events_for_tokenization,
                conversion_function,
                n_jobs=df_conversion_n_jobs,
            )
```

with:

```python
            validation_records = list(iter_converted_tuples(validation_events_for_tokenization, conversion_function, log_every=100))
            del validation_events_for_tokenization
            del validation_full_constants, validation_full_events
            gc.collect()
            log_memory_usage("after-validation-string-conversion-and-raw-validation-free")
```

- [ ] **Step 6: Replace old preprocess calls with converted-record preprocessing**

Replace:

```python
            training_dataset = dp.preprocess_dataset(training_input_strings, training_target_strings, tokenize=tokenize)

            # Tokenize validation dataset
            validation_dataset = dp.preprocess_dataset(validation_input_strings, validation_target_strings, tokenize=tokenize)
```

with:

```python
            training_dataset = dp.preprocess_converted_records(training_records, tokenize=tokenize, keep_text_columns=False)
            del training_records
            gc.collect()
            log_memory_usage("after-training-tokenization-and-record-free")

            validation_dataset = dp.preprocess_converted_records(validation_records, tokenize=tokenize, keep_text_columns=False)
            del validation_records
            gc.collect()
            log_memory_usage("after-validation-tokenization-and-record-free")
```

- [ ] **Step 7: Verify syntax**

Run:

```bash
python -m compileall pipeline 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row
```

Expected: compile completes without `SyntaxError`.

- [ ] **Step 8: Run focused tests**

Run:

```bash
pytest tests/test_df_conversion_helpers.py tests/test_data_processor_streaming.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit MIMIC experiment wiring**

Run:

```bash
git add 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py
git commit -m "Free MIMIC raw DataFrames before model setup

Constraint: The killed run died during DF-to-string conversion before model setup, so CPU memory must be lowered before GPUs matter.
Rejected: Only adding more GPUs | preprocessing is CPU-side and the existing script already requested two GPUs without using distributed training.
Confidence: medium
Scope-risk: moderate
Directive: If logs still show memory growth after conversion, move Task 5 disk-backed Arrow conversion from optional to required.
Tested: python -m compileall pipeline 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row; pytest tests/test_df_conversion_helpers.py tests/test_data_processor_streaming.py -v
Not-tested: Full MIMIC Slurm run"
```

---

### Task 5: Optional fallback if Task 4 still exceeds RAM: write converted records to disk-backed Arrow

**Files:**
- Modify: `pipeline/DFConversionHelpers.py`
- Modify: `pipeline/data_processors/DataProcessorBiomistral.py`
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
- Create: `tests/test_df_conversion_arrow_cache.py`

- [ ] **Step 1: Trigger condition**

Only execute this task if the Task 4 Slurm log still reaches DF conversion and dies before `Setting up model`, or if memory logs exceed 120 GB before tokenization.

- [ ] **Step 2: Add a test for disk-backed conversion cache**

Create `tests/test_df_conversion_arrow_cache.py` with this exact content:

```python
from datasets import load_from_disk

from pipeline.DFConversionHelpers import write_converted_records_to_disk


def test_write_converted_records_to_disk_round_trips(tmp_path):
    def convert(value):
        return f"input-{value}", f"target-{value}", {"idx": value}

    output_dir = tmp_path / "converted"
    dataset = write_converted_records_to_disk([(1,), (2,)], convert, output_dir, log_every=1)

    assert len(dataset) == 2
    assert dataset[0]["input_text"] == "input-1"
    assert dataset[1]["target_text"] == "target-2"

    reloaded = load_from_disk(str(output_dir))
    assert len(reloaded) == 2
    assert reloaded[0]["meta_data"] == {"idx": 1}
```

- [ ] **Step 3: Run test and verify failure before implementation**

Run:

```bash
pytest tests/test_df_conversion_arrow_cache.py -v
```

Expected before implementation: import fails because `write_converted_records_to_disk` does not exist.

- [ ] **Step 4: Implement disk-backed conversion writer**

Add these imports to `pipeline/DFConversionHelpers.py`:

```python
from pathlib import Path
from datasets import Dataset
```

Add this function at the end of `pipeline/DFConversionHelpers.py`:

```python
def write_converted_records_to_disk(list_of_data_tuples, conversion_function, output_dir, log_every=100):
    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = iter_converted_tuples(list_of_data_tuples, conversion_function, log_every=log_every)
    dataset = Dataset.from_generator(lambda: records)
    dataset.save_to_disk(str(output_path))
    log_memory_usage(f"after-saving-converted-records-{output_path}")
    return dataset
```

- [ ] **Step 5: Run cache test**

Run:

```bash
pytest tests/test_df_conversion_arrow_cache.py -v
```

Expected: test passes.

- [ ] **Step 6: Wire cache paths into the experiment**

In `dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`, use `experiment.data_path` to persist conversion outputs:

```python
            training_cache_path = Path(experiment.data_path) / "training_converted_records"
            validation_cache_path = Path(experiment.data_path) / "validation_converted_records"
            training_records = write_converted_records_to_disk(training_events, conversion_function, training_cache_path, log_every=100)
            validation_records = write_converted_records_to_disk(validation_events_for_tokenization, conversion_function, validation_cache_path, log_every=100)
```

Also import `write_converted_records_to_disk` from `pipeline.DFConversionHelpers`.

- [ ] **Step 7: Verify fallback tests**

Run:

```bash
pytest tests/test_df_conversion_helpers.py tests/test_data_processor_streaming.py tests/test_df_conversion_arrow_cache.py -v
python -m compileall pipeline 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row
```

Expected: pytest passes and compileall completes without `SyntaxError`.

- [ ] **Step 8: Commit disk-backed fallback**

Run:

```bash
git add pipeline/DFConversionHelpers.py pipeline/data_processors/DataProcessorBiomistral.py 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py tests/test_df_conversion_arrow_cache.py
git commit -m "Persist converted MIMIC text records to Arrow cache

Constraint: Full MIMIC preprocessing can exceed RAM when raw DataFrames and converted strings coexist.
Rejected: Raising Slurm memory indefinitely | disk-backed datasets fix the structural duplication.
Confidence: medium
Scope-risk: moderate
Directive: Keep cache paths under the experiment data directory so reruns remain reproducible and isolated.
Tested: pytest tests/test_df_conversion_helpers.py tests/test_data_processor_streaming.py tests/test_df_conversion_arrow_cache.py -v; python -m compileall pipeline 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row
Not-tested: Full MIMIC Slurm run"
```

---

### Task 6: Submit and monitor the fixed run

**Files:**
- No code changes unless Slurm output identifies a new blocker.

- [ ] **Step 1: Submit the two-L40S job after Task 1 at minimum**

Run:

```bash
sbatch job/submit_mimic_dora_resume1395_to4185.sh
```

Expected: Slurm prints `Submitted batch job <job_id>`.

- [ ] **Step 2: Monitor early log milestones**

Replace `<job_id>` with the Slurm ID from Step 1 and run:

```bash
tail -f logs/mimic_dora_resume1395_to4185_<job_id>.out
```

Expected milestones:
- `Distributed training: DeepSpeed ZeRO-3 (2 processes)`
- `Training mode: DoRA (standard PEFT)`
- `Converting DFs to Strings: 100 / 44667` and later progress updates
- `after-training-string-conversion-and-raw-train-free`
- `Setting up model`
- `Start training`

- [ ] **Step 3: If it dies before `Setting up model`, capture evidence**

Run:

```bash
tail -n 120 logs/mimic_dora_resume1395_to4185_<job_id>.out
tail -n 120 logs/mimic_dora_resume1395_to4185_<job_id>.err
```

Expected if Task 5 is needed: logs show memory usage rising above 120 GB or another `Killed` during conversion.

- [ ] **Step 4: If it reaches `Start training`, stop changing preprocessing**

Expected: no further root-fix work is needed for the conversion blocker. Continue monitoring training checkpoints only.

## Self-review

Spec coverage:
- Direct use of two L40S: Task 1 modifies the existing resume script in place so the two reserved L40S GPUs on node 201 are used by two distributed processes; Task 6 submits the same existing script.
- Avoiding repeated DF conversion failure: Tasks 2-4 add streaming conversion, progress/memory logs, and earlier raw DataFrame cleanup; Task 5 adds disk-backed fallback if memory remains too high.
- Root cause explanation: Evidence section documents that the failure is CPU preprocessing before model setup, so GPU count alone cannot be the root fix.

Placeholder scan:
- No placeholder-marker text or unspecified test/error handling remains.
- Each code-changing step includes concrete code or exact replacement snippets.

Type/signature consistency:
- `iter_converted_tuples()` yields dictionaries with `input_text`, `target_text`, and `meta_data`.
- `preprocess_converted_records()` accepts those dictionaries and removes `meta_data` before tokenization.
- Experiment wiring uses `training_records` and `validation_records` consistently.
