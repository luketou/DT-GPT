# Bound DF Conversion Workers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent MIMIC DF-to-string preprocessing from being killed by RAM OOM when Slurm CPU count increases.

**Architecture:** Make DF conversion worker count explicit and bounded instead of defaulting to `joblib.Parallel(n_jobs=-1)`. Add a small resolver in `pipeline/DFConversionHelpers.py`, expose it through the MIMIC training CLI, and pass a safe shell default from `job/submit_mimic_dora.sh`.

**Tech Stack:** Python stdlib `os`, `unittest`, `unittest.mock`; existing `joblib.Parallel`; existing MIMIC experiment scripts.

---

## File Structure

- Modify: `pipeline/DFConversionHelpers.py`
  - Add `resolve_df_conversion_n_jobs()` that reads `DTGPT_DF_CONVERSION_N_JOBS` and defaults to `1`.
  - Change `process_all_tuples_multiprocessing()` so `n_jobs=None` resolves through that helper.
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py`
  - Add `--df-conversion-n-jobs` CLI flag.
  - Forward it into the experiment `run()` call.
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
  - Add `df_conversion_n_jobs` parameter to `run()`.
  - Pass it to both training and validation DF-to-string conversion calls.
- Modify: `job/submit_mimic_dora.sh`
  - Add safe default `DTGPT_DF_CONVERSION_N_JOBS=1`.
  - Echo the chosen setting and pass it to Python.
- Create: `tests/test_df_conversion_helpers.py`
  - Unit tests for env/default/invalid worker-count resolution and actual `Parallel(n_jobs=...)` wiring.

### Task 1: Add regression tests for bounded worker resolution

**Files:**
- Create: `tests/test_df_conversion_helpers.py`

- [ ] **Step 1: Write the failing tests**

```python
import os
import unittest
from unittest.mock import patch

from pipeline import DFConversionHelpers as helpers


class TestDFConversionWorkerResolution(unittest.TestCase):
    def test_defaults_to_single_worker_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(helpers.resolve_df_conversion_n_jobs(), 1)

    def test_uses_positive_integer_from_environment(self):
        with patch.dict(os.environ, {"DTGPT_DF_CONVERSION_N_JOBS": "2"}, clear=True):
            self.assertEqual(helpers.resolve_df_conversion_n_jobs(), 2)

    def test_rejects_zero_from_environment(self):
        with patch.dict(os.environ, {"DTGPT_DF_CONVERSION_N_JOBS": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "DTGPT_DF_CONVERSION_N_JOBS"):
                helpers.resolve_df_conversion_n_jobs()

    def test_explicit_argument_overrides_environment(self):
        with patch.dict(os.environ, {"DTGPT_DF_CONVERSION_N_JOBS": "4"}, clear=True):
            self.assertEqual(helpers.resolve_df_conversion_n_jobs(2), 2)


class _FakeParallel:
    last_n_jobs = None

    def __init__(self, n_jobs):
        _FakeParallel.last_n_jobs = n_jobs

    def __call__(self, delayed_calls):
        return [call() for call in delayed_calls]


class _FakeDelayedCall:
    def __init__(self, func, args):
        self.func = func
        self.args = args

    def __call__(self):
        return self.func(*self.args)


def _fake_delayed(func):
    def wrapper(*args):
        return _FakeDelayedCall(func, args)
    return wrapper


def _convert(value):
    return f"input-{value}", f"target-{value}", {"value": value}


class TestProcessAllTuplesMultiprocessing(unittest.TestCase):
    def test_passes_resolved_worker_count_to_joblib(self):
        _FakeParallel.last_n_jobs = None
        with patch.dict(os.environ, {"DTGPT_DF_CONVERSION_N_JOBS": "2"}, clear=True):
            with patch.object(helpers, "Parallel", _FakeParallel), patch.object(helpers, "delayed", _fake_delayed):
                inputs, targets, metas = helpers.process_all_tuples_multiprocessing([(1,), (2,)], _convert)

        self.assertEqual(_FakeParallel.last_n_jobs, 2)
        self.assertEqual(inputs, ("input-1", "input-2"))
        self.assertEqual(targets, ("target-1", "target-2"))
        self.assertEqual(metas, ({"value": 1}, {"value": 2}))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_df_conversion_helpers -v`

Expected: FAIL/ERROR because `resolve_df_conversion_n_jobs` does not exist yet.

### Task 2: Implement bounded worker resolution

**Files:**
- Modify: `pipeline/DFConversionHelpers.py`

- [ ] **Step 1: Add resolver and safe default**

```python
import os


def resolve_df_conversion_n_jobs(n_jobs=None):
    if n_jobs is not None:
        resolved_n_jobs = int(n_jobs)
    else:
        resolved_n_jobs = int(os.environ.get("DTGPT_DF_CONVERSION_N_JOBS", "1"))

    if resolved_n_jobs < 1:
        raise ValueError("DTGPT_DF_CONVERSION_N_JOBS must be a positive integer.")

    return resolved_n_jobs
```

- [ ] **Step 2: Wire resolver into multiprocessing helper**

```python
def process_all_tuples_multiprocessing(list_of_data_tuples, conversion_function, n_jobs=None):
    resolved_n_jobs = resolve_df_conversion_n_jobs(n_jobs)
    logging.info(
        "Converting DFs to Strings with joblib workers: %s for %s tuples",
        resolved_n_jobs,
        len(list_of_data_tuples),
    )
    results = Parallel(n_jobs=resolved_n_jobs)(delayed(conversion_function)(*i) for i in list_of_data_tuples)
    list_input_strings, list_target_strings, list_meta_data = zip(*results)
    return list_input_strings, list_target_strings, list_meta_data
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `python -m unittest tests.test_df_conversion_helpers -v`

Expected: all 5 tests pass.

### Task 3: Expose worker count in the MIMIC training script and Slurm wrapper

**Files:**
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py`
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
- Modify: `job/submit_mimic_dora.sh`

- [ ] **Step 1: Add CLI argument**

```python
parser.add_argument(
    "--df-conversion-n-jobs",
    type=int,
    default=None,
    help="Number of joblib workers for DF-to-string conversion. Defaults to DTGPT_DF_CONVERSION_N_JOBS or 1.",
)
```

- [ ] **Step 2: Forward argument into experiment.run**

```python
df_conversion_n_jobs=args.df_conversion_n_jobs,
```

- [ ] **Step 3: Add experiment parameter and pass it to both conversion calls**

```python
df_conversion_n_jobs=None,
```

```python
training_input_strings, training_target_strings, training_meta_data = process_all_tuples_multiprocessing(
    training_events,
    conversion_function,
    n_jobs=df_conversion_n_jobs,
)
```

```python
validation_input_strings, validation_target_strings, validation_meta_data = process_all_tuples_multiprocessing(
    validation_events_for_tokenization,
    conversion_function,
    n_jobs=df_conversion_n_jobs,
)
```

- [ ] **Step 4: Add Slurm environment default and pass-through**

```bash
DF_CONVERSION_N_JOBS="${DTGPT_DF_CONVERSION_N_JOBS:-1}"
echo "DF conversion joblib workers: ${DF_CONVERSION_N_JOBS}"
--df-conversion-n-jobs "${DF_CONVERSION_N_JOBS}"
```

- [ ] **Step 5: Verify syntax**

Run: `python -m compileall pipeline 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row job`

Expected: compile completes with exit code 0.

### Task 4: Final verification

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run targeted tests**

Run: `python -m unittest tests.test_df_conversion_helpers -v`

Expected: all tests pass.

- [ ] **Step 2: Run lightweight repo syntax check**

Run: `python -m compileall pipeline 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row job`

Expected: exit code 0.

- [ ] **Step 3: Inspect changed files**

Run: `git diff -- pipeline/DFConversionHelpers.py 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py job/submit_mimic_dora.sh tests/test_df_conversion_helpers.py`

Expected: diff only introduces bounded worker configuration and tests.

## Self-Review

- Spec coverage: The plan addresses the observed RAM OOM failure mode by removing `n_jobs=-1`, making worker count explicit, and preserving an override for controlled CPU/RAM tradeoffs.
- Placeholder scan: No placeholders, TBDs, or vague test instructions remain.
- Type consistency: The worker-count parameter is consistently named `df_conversion_n_jobs` in Python and `DTGPT_DF_CONVERSION_N_JOBS` / `DF_CONVERSION_N_JOBS` in shell.
