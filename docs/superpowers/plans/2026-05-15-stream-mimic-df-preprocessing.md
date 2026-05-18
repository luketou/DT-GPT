# Stream MIMIC DF Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `job/submit_mimic_dora_resume1395_to2100.sh` from being killed during MIMIC DF-to-string preprocessing by removing the current full-dataset in-memory materialization path.

**Architecture:** Replace the current “convert everything into giant Python lists, then call `Dataset.from_dict`” path with a streaming/chunked pipeline. Keep the old helper as a compatibility wrapper, add a batch iterator in `pipeline/DFConversionHelpers.py`, teach `DataProcessorBiomistral` to build/tokenize a dataset from a generator, and refactor the MIMIC training script to stream normalized tuples directly into Hugging Face datasets instead of holding all converted text in RAM at once. Add a preprocessing-only stop flag so the fix can be validated without starting model training.

**Tech Stack:** Python stdlib `itertools`, `unittest`, `unittest.mock`; existing `joblib`; Hugging Face `datasets.Dataset`; existing MIMIC experiment and Slurm scripts.

---

## File Structure

- Modify: `pipeline/DFConversionHelpers.py`
  - Add chunk validation and a batch iterator that yields converted `(input_text, target_text, meta_data)` rows in bounded chunks.
  - Keep `process_all_tuples_multiprocessing()` for backward compatibility, but re-implement it on top of the iterator.
- Modify: `pipeline/data_processors/DataProcessorBiomistral.py`
  - Refactor tokenization so it can operate on a `Dataset` built from a generator instead of requiring two full Python lists.
  - Add `preprocess_dataset_from_generator()`.
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
  - Replace the full-list preprocessing flow with streaming normalization → chunked DF conversion → generator-backed dataset creation.
  - Add a preprocessing-only stop flag and chunk-size plumbing.
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py`
  - Expose `--df-conversion-chunk-size` and `--stop-after-preprocessing`.
- Modify: `job/submit_mimic_dora.sh`
  - Read and print safe defaults for new preprocessing controls.
- Modify: `job/submit_mimic_dora_resume1395_to2100.sh`
  - Surface the same environment knobs for the failing resume workflow.
- Create: `tests/test_streaming_mimic_preprocessing.py`
  - Regression tests for bounded chunk iteration and generator-backed dataset preprocessing.

### Task 1: Add regression tests for bounded conversion and generator-backed tokenization

**Files:**
- Create: `tests/test_streaming_mimic_preprocessing.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest
from unittest.mock import patch

from pipeline import DFConversionHelpers as helpers
from pipeline.data_processors.DataProcessorBiomistral import DataProcessorBiomistral


class _FakeTokenizer:
    def __call__(self, text, max_length, truncation):
        return {
            "input_ids": [[idx, idx + 1] for idx, _ in enumerate(text)],
            "attention_mask": [[1, 1] for _ in text],
        }


class TestDFConversionChunking(unittest.TestCase):
    def test_iter_converted_tuple_batches_preserves_order_and_chunking(self):
        tuples = [(0,), (1,), (2,), (3,), (4,)]

        def convert(value):
            return f"input-{value}", f"target-{value}", {"value": value}

        batches = list(
            helpers.iter_converted_tuple_batches(
                tuples,
                convert,
                n_jobs=1,
                chunk_size=2,
            )
        )

        flattened = [row for batch in batches for row in batch]
        self.assertEqual([[item[2]["value"] for item in batch] for batch in batches], [[0, 1], [2, 3], [4]])
        self.assertEqual([item[0] for item in flattened], ["input-0", "input-1", "input-2", "input-3", "input-4"])
        self.assertEqual([item[1] for item in flattened], ["target-0", "target-1", "target-2", "target-3", "target-4"])

    def test_iter_converted_tuple_batches_rejects_non_positive_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            list(helpers.iter_converted_tuple_batches([(1,)], lambda value: value, chunk_size=0))


class TestDataProcessorStreaming(unittest.TestCase):
    def make_processor(self):
        processor = DataProcessorBiomistral.__new__(DataProcessorBiomistral)
        processor.tokenizer = _FakeTokenizer()
        processor.max_total_length = 64
        processor.collator_setting = "completion"
        processor.response_template = "<patient_prediction>"
        return processor

    def test_preprocess_dataset_from_generator_builds_dataset_without_full_lists(self):
        processor = self.make_processor()

        def row_factory():
            yield ("history-a", "target-a")
            yield ("history-b", "target-b")

        dataset = processor.preprocess_dataset_from_generator(
            row_factory,
            tokenize=False,
            map_batch_size=2,
        )

        self.assertEqual(dataset.num_rows, 2)
        self.assertIn("concatenated_text", dataset.column_names)
        self.assertEqual(dataset[0]["input_text"], "history-a <patient_prediction>")
        self.assertEqual(dataset[1]["target_text"], "target-b")

    def test_preprocess_dataset_from_generator_tokenizes_batches(self):
        processor = self.make_processor()

        def row_factory():
            yield ("history-a", "target-a")
            yield ("history-b", "target-b")

        dataset = processor.preprocess_dataset_from_generator(
            row_factory,
            tokenize=True,
            map_batch_size=2,
        )

        self.assertIn("input_ids", dataset.column_names)
        self.assertEqual(dataset.num_rows, 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_streaming_mimic_preprocessing -v`

Expected: FAIL because `iter_converted_tuple_batches()` and `preprocess_dataset_from_generator()` do not exist yet.

### Task 2: Implement chunked DF conversion helpers

**Files:**
- Modify: `pipeline/DFConversionHelpers.py`

- [ ] **Step 1: Add chunk-size resolver and chunk iterator helpers**

```python
import logging
import os
from joblib import Parallel, delayed


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


def resolve_df_conversion_chunk_size(chunk_size=None):
    try:
        if chunk_size is not None:
            resolved_chunk_size = int(chunk_size)
        else:
            resolved_chunk_size = int(os.environ.get("DTGPT_DF_CONVERSION_CHUNK_SIZE", "128"))
    except ValueError as error:
        raise ValueError("DTGPT_DF_CONVERSION_CHUNK_SIZE must be a positive integer.") from error

    if resolved_chunk_size < 1:
        raise ValueError("DTGPT_DF_CONVERSION_CHUNK_SIZE must be a positive integer.")

    return resolved_chunk_size


def iter_tuple_chunks(list_of_data_tuples, chunk_size):
    resolved_chunk_size = resolve_df_conversion_chunk_size(chunk_size)
    current_chunk = []

    for data_tuple in list_of_data_tuples:
        current_chunk.append(data_tuple)
        if len(current_chunk) == resolved_chunk_size:
            yield current_chunk
            current_chunk = []

    if current_chunk:
        yield current_chunk
```

- [ ] **Step 2: Add batch conversion iterator with bounded memory use**

```python
def iter_converted_tuple_batches(list_of_data_tuples, conversion_function, n_jobs=None, chunk_size=None):
    resolved_n_jobs = resolve_df_conversion_n_jobs(n_jobs)
    resolved_chunk_size = resolve_df_conversion_chunk_size(chunk_size)
    total_tuples = len(list_of_data_tuples)

    logging.info(
        "Streaming DF conversion with joblib workers: %s | chunk size: %s | tuples: %s",
        resolved_n_jobs,
        resolved_chunk_size,
        total_tuples,
    )

    processed = 0
    for chunk_index, data_chunk in enumerate(iter_tuple_chunks(list_of_data_tuples, resolved_chunk_size), start=1):
        if resolved_n_jobs == 1:
            converted_chunk = [conversion_function(*data_tuple) for data_tuple in data_chunk]
        else:
            converted_chunk = Parallel(n_jobs=resolved_n_jobs)(
                delayed(conversion_function)(*data_tuple) for data_tuple in data_chunk
            )

        processed += len(converted_chunk)
        logging.info(
            "Converted DF chunk %s | chunk rows: %s | processed: %s / %s",
            chunk_index,
            len(converted_chunk),
            processed,
            total_tuples,
        )
        yield converted_chunk
```

- [ ] **Step 3: Keep the old API but re-implement it on top of the iterator**

```python
def process_all_tuples_multiprocessing(list_of_data_tuples, conversion_function, n_jobs=None, chunk_size=None):
    list_input_strings = []
    list_target_strings = []
    list_meta_data = []

    for converted_chunk in iter_converted_tuple_batches(
        list_of_data_tuples,
        conversion_function,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    ):
        if not converted_chunk:
            continue

        chunk_inputs, chunk_targets, chunk_meta = zip(*converted_chunk)
        list_input_strings.extend(chunk_inputs)
        list_target_strings.extend(chunk_targets)
        list_meta_data.extend(chunk_meta)

    return list_input_strings, list_target_strings, list_meta_data
```

- [ ] **Step 4: Run targeted tests**

Run: `python -m unittest tests.test_streaming_mimic_preprocessing.TestDFConversionChunking -v`

Expected: PASS for the chunking tests.

### Task 3: Refactor `DataProcessorBiomistral` to support generator-backed datasets

**Files:**
- Modify: `pipeline/data_processors/DataProcessorBiomistral.py`

- [ ] **Step 1: Extract tokenization to a dataset-level helper**

```python
from datasets import Dataset


class DataProcessorBiomistral():
    ...

    def preprocess_dataset(self, list_of_input_strings, list_of_target_strings, tokenize=True, map_batch_size=128):
        dataset = Dataset.from_dict({
            "input_text": list_of_input_strings,
            "target_text": list_of_target_strings,
        })
        return self._tokenize_dataset(dataset, tokenize=tokenize, map_batch_size=map_batch_size)
```

- [ ] **Step 2: Add generator-backed dataset preprocessing**

```python
    def preprocess_dataset_from_generator(self, row_factory, tokenize=True, map_batch_size=128):
        def dataset_rows():
            for input_text, target_text in row_factory():
                yield {
                    "input_text": input_text,
                    "target_text": target_text,
                }

        dataset = Dataset.from_generator(dataset_rows)
        return self._tokenize_dataset(dataset, tokenize=tokenize, map_batch_size=map_batch_size)
```

- [ ] **Step 3: Make `_tokenize_dataset()` preprocess batches instead of whole Python lists**

```python
    def _tokenize_dataset(self, dataset, tokenize, map_batch_size):
        logging.info("DataProcessor: tokenizing dataset!")

        def preprocess_function(samples):
            inputs = self.preprocess_inputs(samples["input_text"])
            targets = self.preprocess_outputs(samples["target_text"])
            concatenated_text = [f"{inputs[idx]} {targets[idx]}" for idx in range(len(inputs))]

            if tokenize:
                model_inputs = self.tokenizer(
                    text=concatenated_text,
                    max_length=self.max_total_length,
                    truncation=True,
                )
            else:
                model_inputs = {}

            model_inputs["input_text"] = inputs
            model_inputs["target_text"] = targets
            model_inputs["concatenated_text"] = concatenated_text
            return model_inputs

        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=map_batch_size,
        )
        logging.info("DataProcessor: finished tokenizing dataset!")
        return tokenized_dataset
```

- [ ] **Step 4: Run targeted tests**

Run: `python -m unittest tests.test_streaming_mimic_preprocessing.TestDataProcessorStreaming -v`

Expected: PASS for the streaming dataset tests.

### Task 4: Stream normalization and DF conversion inside the MIMIC training script

**Files:**
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`

- [ ] **Step 1: Update imports to use the streaming helper**

```python
from pipeline.DFConversionHelpers import iter_converted_tuple_batches
import gc
```

- [ ] **Step 2: Extend `run()` with chunk-size and preprocessing-only controls**

```python
    def run(
        self,
        ...,
        sft_dataset_num_proc=1,
        df_conversion_n_jobs=None,
        df_conversion_chunk_size=128,
        dataset_map_batch_size=128,
        stop_after_preprocessing=False,
        ...,
    ):
```

- [ ] **Step 3: Replace the current two-list training preprocessing flow with streaming factories**

```python
        def build_conversion_tuple_stream(events_for_split, normalization_filter):
            for curr_const_row, true_events_input, true_future_events_input, target_dataframe in events_for_split:
                yield (
                    column_mapping,
                    curr_const_row,
                    normalization_filter.normalize_and_filter(
                        true_events_input.copy(),
                        None,
                        replace_nan_rows=False,
                        replace_missing_in_prediction=False,
                        verbose=False,
                        specific_column_list=["lab_26499_4"],
                    )[0],
                    true_future_events_input,
                    normalization_filter.normalize_and_filter(
                        target_dataframe.copy(),
                        None,
                        replace_nan_rows=False,
                        replace_missing_in_prediction=False,
                        verbose=False,
                    )[0],
                    filtering_rows_rest_budget,
                    SEQUENCE_MAX_LENGTH_IN_TOKENS,
                    DECIMAL_PRECISION,
                    prompt,
                    constant_column_mapping,
                )

        def build_text_pair_factory(events_for_split, normalization_filter):
            conversion_tuples = list(build_conversion_tuple_stream(events_for_split, normalization_filter))

            def row_factory():
                for converted_chunk in iter_converted_tuple_batches(
                    conversion_tuples,
                    conversion_function,
                    n_jobs=df_conversion_n_jobs,
                    chunk_size=df_conversion_chunk_size,
                ):
                    for input_text, target_text, _meta_data in converted_chunk:
                        yield input_text, target_text

            return conversion_tuples, row_factory
```

- [ ] **Step 4: Remove the full `training_input_strings` / `validation_input_strings` materialization path**

```python
            training_norm_filter = Only_Double3_sigma_Filtering(path_to_statistics_file)
            training_conversion_tuples, training_row_factory = build_text_pair_factory(
                training_events,
                training_norm_filter,
            )

            preview_batch = next(
                iter_converted_tuple_batches(
                    training_conversion_tuples[:2],
                    conversion_function,
                    n_jobs=1,
                    chunk_size=2,
                )
            )
            logging.info("Example of input: %s", preview_batch[0][0])
            logging.info("Example of target: %s", preview_batch[0][1])
            if len(preview_batch) > 1:
                logging.info("Example of input: %s", preview_batch[1][0])
                logging.info("Example of target: %s", preview_batch[1][1])

            tokenize = True
            training_dataset = dp.preprocess_dataset_from_generator(
                training_row_factory,
                tokenize=tokenize,
                map_batch_size=dataset_map_batch_size,
            )
            del training_conversion_tuples
            del training_events
            gc.collect()

            validation_conversion_tuples, validation_row_factory = build_text_pair_factory(
                validation_events,
                training_norm_filter,
            )
            validation_dataset = dp.preprocess_dataset_from_generator(
                validation_row_factory,
                tokenize=tokenize,
                map_batch_size=dataset_map_batch_size,
            )
            del validation_conversion_tuples
            del validation_events
            gc.collect()

            if stop_after_preprocessing:
                logging.info("Stopping after preprocessing as requested.")
                return
```

- [ ] **Step 5: Tighten the memory profile further by not prebuilding a second normalized tuple list**

Replace the `list(...)` call from Step 3 with a true factory that recreates the stream on demand:

```python
        def build_text_pair_factory(events_for_split, normalization_filter):
            def conversion_tuple_factory():
                return build_conversion_tuple_stream(events_for_split, normalization_filter)

            def row_factory():
                conversion_tuples = conversion_tuple_factory()
                tuple_count = len(events_for_split)
                for converted_chunk in iter_converted_tuple_batches(
                    conversion_tuples,
                    conversion_function,
                    n_jobs=df_conversion_n_jobs,
                    chunk_size=df_conversion_chunk_size,
                    total_tuples=tuple_count,
                ):
                    for input_text, target_text, _meta_data in converted_chunk:
                        yield input_text, target_text

            return row_factory
```

Also update `iter_converted_tuple_batches()` to accept an optional `total_tuples` argument so logging still works when the input is a generator.

- [ ] **Step 6: Run a syntax check on the training files**

Run: `python -m compileall pipeline 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row`

Expected: exit code 0.

### Task 5: Expose the new preprocessing controls in CLI and Slurm wrappers

**Files:**
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py`
- Modify: `job/submit_mimic_dora.sh`
- Modify: `job/submit_mimic_dora_resume1395_to2100.sh`

- [ ] **Step 1: Add CLI flags for chunk size, map batch size, and preprocessing-only stop**

```python
    parser.add_argument(
        "--df-conversion-chunk-size",
        type=int,
        default=128,
        help="Bound the number of converted DF samples held in memory per preprocessing chunk.",
    )
    parser.add_argument(
        "--dataset-map-batch-size",
        type=int,
        default=128,
        help="Batch size for Dataset.map during tokenization.",
    )
    parser.add_argument(
        "--stop-after-preprocessing",
        action="store_true",
        help="Build/tokenize datasets and exit before model setup/training.",
    )
```

- [ ] **Step 2: Forward the new args into `experiment.run()`**

```python
        df_conversion_chunk_size=args.df_conversion_chunk_size,
        dataset_map_batch_size=args.dataset_map_batch_size,
        stop_after_preprocessing=args.stop_after_preprocessing,
```

- [ ] **Step 3: Add shell defaults and pass-through in `job/submit_mimic_dora.sh`**

```bash
DATASET_MAP_BATCH_SIZE="${DTGPT_DATASET_MAP_BATCH_SIZE:-128}"
DF_CONVERSION_CHUNK_SIZE="${DTGPT_DF_CONVERSION_CHUNK_SIZE:-128}"
STOP_AFTER_PREPROCESSING="${DTGPT_STOP_AFTER_PREPROCESSING:-0}"

echo "DF conversion chunk size: ${DF_CONVERSION_CHUNK_SIZE}"
echo "Dataset.map batch size: ${DATASET_MAP_BATCH_SIZE}"
echo "Stop after preprocessing: ${STOP_AFTER_PREPROCESSING}"

STOP_AFTER_PREPROCESSING_FLAG=()
if [ "${STOP_AFTER_PREPROCESSING}" = "1" ]; then
    STOP_AFTER_PREPROCESSING_FLAG=(--stop-after-preprocessing)
fi
```

Then extend the Python command:

```bash
        --df-conversion-n-jobs "${DF_CONVERSION_N_JOBS}" \
        --df-conversion-chunk-size "${DF_CONVERSION_CHUNK_SIZE}" \
        --dataset-map-batch-size "${DATASET_MAP_BATCH_SIZE}" \
        "${STOP_AFTER_PREPROCESSING_FLAG[@]}"
```

- [ ] **Step 4: Surface the same defaults in the failing resume wrapper**

```bash
export DTGPT_DF_CONVERSION_CHUNK_SIZE="${DTGPT_DF_CONVERSION_CHUNK_SIZE:-128}"
export DTGPT_DATASET_MAP_BATCH_SIZE="${DTGPT_DATASET_MAP_BATCH_SIZE:-128}"
export DTGPT_STOP_AFTER_PREPROCESSING="${DTGPT_STOP_AFTER_PREPROCESSING:-0}"
```

- [ ] **Step 5: Verify syntax**

Run: `python -m compileall job 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row`

Expected: exit code 0.

### Task 6: Verify the fix with a preprocessing-only run before full resume training

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run unit tests**

Run: `python -m unittest tests.test_streaming_mimic_preprocessing -v`

Expected: all tests pass.

- [ ] **Step 2: Run lightweight syntax validation**

Run: `python -m compileall pipeline job 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row`

Expected: exit code 0.

- [ ] **Step 3: Run the existing local setup smoke check**

Run: `python 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/smoke_check_mimic_local_setup.py`

Expected: prints `MIMIC local setup smoke check passed.`

- [ ] **Step 4: Run a preprocessing-only Slurm-compatible smoke command with a tiny split**

Run:

```bash
DTGPT_PATIENT_SPLIT_FRACTION=0.02 \
DTGPT_NUM_SAMPLES_TO_GENERATE=1 \
DTGPT_MAX_NEW_TOKENS=32 \
DTGPT_MAX_STEPS=1 \
DTGPT_DF_CONVERSION_N_JOBS=1 \
DTGPT_DF_CONVERSION_CHUNK_SIZE=64 \
DTGPT_DATASET_MAP_BATCH_SIZE=64 \
DTGPT_STOP_AFTER_PREPROCESSING=1 \
bash job/submit_mimic_dora_resume1395_to2100.sh
```

Expected: log reaches `Stopping after preprocessing as requested.` without `Killed`.

- [ ] **Step 5: Run the real resume job with bounded preprocessing**

Run:

```bash
DTGPT_DF_CONVERSION_N_JOBS=1 \
DTGPT_DF_CONVERSION_CHUNK_SIZE=128 \
DTGPT_DATASET_MAP_BATCH_SIZE=128 \
DTGPT_STOP_AFTER_PREPROCESSING=0 \
bash job/submit_mimic_dora_resume1395_to2100.sh
```

Expected: job advances past DF conversion, model setup starts, and the log no longer ends at `Converting DFs to Strings with joblib workers: 1 for 44667 tuples` followed by `Killed`.

## Self-Review

- Spec coverage: This plan directly addresses the observed failure point in the May 15, 2026 logs: `job/submit_mimic_dora.sh` is killed after `Converting DFs to Strings with joblib workers: 1 for 44667 tuples`. The fix removes the full in-memory conversion path, adds bounded chunking, and adds a preprocessing-only validation path for the exact resume workflow the user named.
- Placeholder scan: No TBD/TODO placeholders remain. Every task lists exact file paths, code snippets, and commands.
- Type consistency: The Python parameter names are `df_conversion_n_jobs`, `df_conversion_chunk_size`, `dataset_map_batch_size`, and `stop_after_preprocessing`; the shell environment names are `DTGPT_DF_CONVERSION_N_JOBS`, `DTGPT_DF_CONVERSION_CHUNK_SIZE`, `DTGPT_DATASET_MAP_BATCH_SIZE`, and `DTGPT_STOP_AFTER_PREPROCESSING`.
