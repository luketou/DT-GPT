# MIMIC Prediction Parser Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the MIMIC BioMistral train/eval path tolerate malformed prediction JSON lengths so `job/submit_mimic_train_v100.sh` no longer crashes with `ValueError: All arrays must be of the same length`, and remove the related parser and merge warnings from batch logs.

**Architecture:** Keep the current training entrypoint and experiment wiring intact, but harden the prediction parsing boundary in `pipeline/data_generators/DataFrameConvertTDBDMIMIC.py` before Pandas sees any model output. Then isolate the numeric sample aggregation in a small helper so the `Experiment` merge step can handle all-NaN prediction cubes without `RuntimeWarning: Mean of empty slice`.

**Tech Stack:** Python 3.8, pandas, numpy, unittest, Hugging Face/TRL training wrappers, Slurm batch jobs

---

## File Map

- `pipeline/data_generators/DataFrameConvertTDBDMIMIC.py`
  Responsibility: turn BioMistral prediction JSON strings into per-patient prediction DataFrames for MIMIC.
- `pipeline/Experiment.py`
  Responsibility: post-process generated samples, merge multiple prediction samples, and fall back to empty predictions on parser failures.
- `pipeline/prediction_aggregation.py`
  Responsibility: contain warning-free numeric aggregation helpers so `Experiment.py` can stay thin and the merge logic can be unit tested in isolation.
- `tests/test_mimic_prediction_parsing.py`
  Responsibility: regression tests for mismatched JSON list lengths, sparse prediction-hour expansion, and exact output horizon length.
- `tests/test_prediction_aggregation.py`
  Responsibility: regression tests for mean aggregation with all-NaN slices and mixed valid/NaN samples.
- `job/submit_mimic_train_v100.sh`
  Responsibility: reproduction entrypoint for the failing workload; do not modify unless validation shows an environment issue rather than a parser issue.

### Task 1: Capture the Current Parser Failure in Regression Tests

**Files:**
- Create: `tests/test_mimic_prediction_parsing.py`
- Modify: `pipeline/data_generators/DataFrameConvertTDBDMIMIC.py:538-636`

- [ ] **Step 1: Write the failing regression tests**

```python
import json
import unittest
import warnings

import pandas as pd

from pipeline.data_generators.DataFrameConvertTDBDMIMIC import (
    DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC,
)


class MimicPredictionParsingTests(unittest.TestCase):
    def setUp(self):
        self.column_name_mapping = pd.DataFrame(
            {
                "descriptive_column_name": [
                    "Heart Rate",
                    "Respiratory Rate",
                ],
                "original_column_names": [
                    "lab_hr",
                    "lab_rr",
                ],
            }
        )
        self.all_prediction_days = pd.to_datetime(
            [
                "2026-04-20 00:00:00",
                "2026-04-20 01:00:00",
                "2026-04-20 02:00:00",
            ]
        )
        self.all_column_names = [
            "lab_hr",
            "lab_rr",
            "date",
            "patientid",
            "patient_sample_index",
        ]

    def test_mismatched_column_lengths_are_normalized_to_prediction_horizon(self):
        prediction = json.dumps(
            {
                "Heart Rate": ["80", "82"],
                "Respiratory Rate": ["18", "19", "20", "21"],
            }
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = (
                DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC.convert_from_strings_to_df(
                    self.column_name_mapping,
                    prediction,
                    self.all_prediction_days,
                    patientid=55,
                    patient_sample_index="split_0",
                    all_column_names=self.all_column_names,
                    all_unk_columns=[],
                )
            )

        self.assertEqual(len(result), 3)
        self.assertEqual(result.loc[0, "lab_hr"], "80")
        self.assertEqual(result.loc[1, "lab_hr"], "82")
        self.assertTrue(pd.isna(result.loc[2, "lab_hr"]))
        self.assertEqual(result["lab_rr"].tolist(), ["18", "19", "20"])
        self.assertFalse(
            any(isinstance(w.message, FutureWarning) for w in caught),
            msg=[str(w.message) for w in caught],
        )

    def test_sparse_prediction_hours_expand_to_the_full_output_horizon(self):
        prediction = json.dumps(
            {
                "Heart Rate": ["80", "84"],
                "Respiratory Rate": ["18", "19", "20"],
            }
        )
        prediction_days_column_wise = {
            "Variables to predict for respective hours": {
                "Heart Rate": [24, 72],
                "Respiratory Rate": [24, 48, 72],
            }
        }

        result = (
            DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC.convert_from_strings_to_df(
                self.column_name_mapping,
                prediction,
                self.all_prediction_days,
                patientid=55,
                patient_sample_index="split_0",
                all_column_names=self.all_column_names,
                all_unk_columns=[],
                prediction_days_column_wise=prediction_days_column_wise,
            )
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(result.loc[0, "lab_hr"], "80")
        self.assertTrue(pd.isna(result.loc[1, "lab_hr"]))
        self.assertEqual(result.loc[2, "lab_hr"], "84")

    def test_missing_prediction_column_becomes_all_nan_instead_of_crashing(self):
        prediction = json.dumps(
            {
                "Respiratory Rate": ["18", "19", "20"],
            }
        )

        result = (
            DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC.convert_from_strings_to_df(
                self.column_name_mapping,
                prediction,
                self.all_prediction_days,
                patientid=55,
                patient_sample_index="split_0",
                all_column_names=self.all_column_names,
                all_unk_columns=[],
            )
        )

        self.assertEqual(len(result), 3)
        self.assertTrue(pd.isna(result.loc[:, "lab_hr"]).all())
        self.assertEqual(result["lab_rr"].tolist(), ["18", "19", "20"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the parser tests to confirm the current failure**

Run: `conda run -n dtgpt python -m unittest tests.test_mimic_prediction_parsing -v`

Expected: FAIL with a traceback ending in `ValueError: All arrays must be of the same length`.

- [ ] **Step 3: Commit the failing test scaffold**

```bash
git add tests/test_mimic_prediction_parsing.py
git commit -m "test: capture mimic prediction parsing regressions"
```

### Task 2: Normalize Prediction Columns Before Pandas Builds the DataFrame

**Files:**
- Modify: `pipeline/data_generators/DataFrameConvertTDBDMIMIC.py:538-636`
- Test: `tests/test_mimic_prediction_parsing.py`

- [ ] **Step 1: Add small helpers for empty-frame construction and column-length normalization**

```python
    def _build_empty_prediction_df(all_column_names, all_prediction_days, patientid, patient_sample_index):
        empty_df = pd.DataFrame(index=range(len(all_prediction_days)), columns=all_column_names)
        empty_df["date"] = pd.to_datetime(all_prediction_days)
        empty_df["patientid"] = patientid
        empty_df["patient_sample_index"] = patient_sample_index
        empty_df = empty_df.reset_index(drop=True)
        return empty_df

    def _normalize_prediction_column_lengths(data_as_dic, expected_columns, expected_length, patientid, patient_sample_index):
        normalized = {}
        mismatches = []

        for column_name in expected_columns:
            raw_values = data_as_dic.get(column_name, [])
            if raw_values is None:
                raw_values = []
            if not isinstance(raw_values, list):
                raise ValueError(
                    "Prediction column "
                    + str(column_name)
                    + " for patient "
                    + str(patientid)
                    + " sample "
                    + str(patient_sample_index)
                    + " did not decode to a JSON list."
                )

            original_length = len(raw_values)
            normalized_values = list(raw_values[:expected_length])
            if len(normalized_values) < expected_length:
                normalized_values.extend([pd.NA] * (expected_length - len(normalized_values)))

            if original_length != expected_length:
                mismatches.append((column_name, original_length, expected_length))

            normalized[column_name] = normalized_values

        if len(mismatches) > 0:
            logging.warning(
                "Normalized malformed prediction lengths for patient %s sample %s: %s",
                patientid,
                patient_sample_index,
                mismatches,
            )

        return normalized
```

- [ ] **Step 2: Replace the fragile `convert_from_strings_to_df` pre-DataFrame path**

```python
    def convert_from_strings_to_df(column_name_mapping, string_output, all_prediction_days, patientid, patient_sample_index,
                                   all_column_names, all_unk_columns, prediction_days_column_wise=None):

        assert "patientid" in all_column_names and "patient_sample_index" in all_column_names and "date" in all_column_names, "DataFrameConverter: PatientID or patient_sample_index or date not in all_column_names!"

        empty_df = DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC._build_empty_prediction_df(
            all_column_names,
            all_prediction_days,
            patientid,
            patient_sample_index,
        )

        try:
            data_as_dic = json.loads(string_output)
            if not isinstance(data_as_dic, dict):
                raise ValueError("Prediction output must decode to a JSON object.")

            if prediction_days_column_wise is not None:
                column_wise_date_mapping = prediction_days_column_wise["Variables to predict for respective hours"]
                all_positions = sorted(list(set([y for x in column_wise_date_mapping.values() for y in x])))

                for col in column_wise_date_mapping.keys():
                    values = list(data_as_dic.get(col, []))
                    positions = column_wise_date_mapping[col]
                    index_lookup = {position: idx for idx, position in enumerate(all_positions)}
                    expanded_values = [pd.NA] * len(all_positions)

                    for source_idx, position in enumerate(positions):
                        if source_idx < len(values):
                            expanded_values[index_lookup[position]] = values[source_idx]

                    data_as_dic[col] = expanded_values

                expected_columns = list(column_wise_date_mapping.keys())
            else:
                expected_columns = sorted(list(data_as_dic.keys()))

            data_as_dic = DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC._normalize_prediction_column_lengths(
                data_as_dic,
                expected_columns,
                len(all_prediction_days),
                patientid,
                patient_sample_index,
            )

            prediction_df = pd.DataFrame.from_dict(data_as_dic, orient="columns")
            prediction_df = prediction_df.reindex(range(len(all_prediction_days)))
            prediction_df.sort_index(inplace=True)

        except Exception as e:
            print("Making empty dataframe due to JSON error for patientid: " + str(patientid) + " and patient_sample_index: " + str(patient_sample_index))
            print(string_output)
            raise e
```

- [ ] **Step 3: Run the parser tests again**

Run: `conda run -n dtgpt python -m unittest tests.test_mimic_prediction_parsing -v`

Expected: PASS for all three tests.

- [ ] **Step 4: Run a syntax pass over the touched module and tests**

Run: `conda run -n dtgpt python -m compileall pipeline/data_generators/DataFrameConvertTDBDMIMIC.py tests/test_mimic_prediction_parsing.py`

Expected: `Compiling ...` lines and no syntax errors.

- [ ] **Step 5: Commit the parser hardening**

```bash
git add pipeline/data_generators/DataFrameConvertTDBDMIMIC.py tests/test_mimic_prediction_parsing.py
git commit -m "fix: normalize malformed mimic prediction lengths"
```

### Task 3: Stop `Experiment` Mean Aggregation from Warning on All-NaN Samples

**Files:**
- Create: `pipeline/prediction_aggregation.py`
- Create: `tests/test_prediction_aggregation.py`
- Modify: `pipeline/Experiment.py:638-656`

- [ ] **Step 1: Write the failing aggregation regression tests**

```python
import unittest
import warnings

import numpy as np

from pipeline.prediction_aggregation import aggregate_prediction_cube


class PredictionAggregationTests(unittest.TestCase):
    def test_mean_aggregation_preserves_nan_without_empty_slice_warning(self):
        cube = np.array(
            [
                [[np.nan, np.nan]],
                [[1.0, np.nan]],
            ],
            dtype=np.float32,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            aggregated = aggregate_prediction_cube(cube, "mean")

        self.assertTrue(np.isnan(aggregated[0, 0]))
        self.assertEqual(aggregated[1, 0], 1.0)
        self.assertFalse(
            any("Mean of empty slice" in str(w.message) for w in caught),
            msg=[str(w.message) for w in caught],
        )

    def test_percentile_strategy_keeps_existing_behavior_for_non_nan_values(self):
        cube = np.array(
            [
                [[2.0, 4.0, 6.0]],
            ],
            dtype=np.float32,
        )

        aggregated = aggregate_prediction_cube(cube, "50th percentile")
        self.assertEqual(aggregated[0, 0], 4.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the aggregation tests to confirm the helper is missing**

Run: `conda run -n dtgpt python -m unittest tests.test_prediction_aggregation -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.prediction_aggregation'`.

- [ ] **Step 3: Create the aggregation helper module**

```python
import numpy as np


def aggregate_prediction_cube(prediction_cube, sample_merging_strategy):
    if sample_merging_strategy == "mean":
        valid_counts = np.sum(~np.isnan(prediction_cube), axis=2)
        value_sums = np.nansum(prediction_cube, axis=2)
        aggregated = np.divide(
            value_sums,
            valid_counts,
            out=np.full(value_sums.shape, np.nan, dtype=np.float32),
            where=valid_counts > 0,
        )
        return aggregated.astype(np.float32)

    if sample_merging_strategy == "50th percentile":
        return np.percentile(prediction_cube, 50, axis=2).astype(np.float32)

    raise Exception("Experiment: unknown sample_merging_strategy provided!")
```

- [ ] **Step 4: Wire `Experiment.py` to use the new helper**

```python
from pipeline.prediction_aggregation import aggregate_prediction_cube


                interesting_df_cols = [np.expand_dims(x[1][target_cols_orginal].values, axis=2) for x in interesting_samples]
                merged_np = np.concatenate(interesting_df_cols, axis=2)
                merged_np = merged_np.astype(np.float32)

                aggregated_np = aggregate_prediction_cube(
                    merged_np,
                    sample_merging_strategy,
                )
```

- [ ] **Step 5: Run the aggregation tests again**

Run: `conda run -n dtgpt python -m unittest tests.test_prediction_aggregation -v`

Expected: PASS for both tests, with no `RuntimeWarning: Mean of empty slice`.

- [ ] **Step 6: Commit the aggregation fix**

```bash
git add pipeline/prediction_aggregation.py pipeline/Experiment.py tests/test_prediction_aggregation.py
git commit -m "fix: avoid empty-slice warnings in prediction merging"
```

### Task 4: Verify the Full MIMIC Recovery Surface

**Files:**
- Modify: `pipeline/data_generators/DataFrameConvertTDBDMIMIC.py:538-636`
- Modify: `pipeline/Experiment.py:638-656`
- Test: `tests/test_mimic_prediction_parsing.py`
- Test: `tests/test_prediction_aggregation.py`

- [ ] **Step 1: Run the focused unit test suite together**

Run: `conda run -n dtgpt python -m unittest tests.test_mimic_prediction_parsing tests.test_prediction_aggregation -v`

Expected: PASS for all tests.

- [ ] **Step 2: Run a repo syntax pass on the affected codepaths**

Run: `conda run -n dtgpt python -m compileall pipeline tests 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row`

Expected: `Compiling ...` lines and no syntax errors.

- [ ] **Step 3: Run the existing environment smoke check used by the Slurm job**

Run: `conda run -n dtgpt python 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/smoke_check_mimic_local_setup.py`

Expected: exit code `0` and environment/model/data-path checks succeed.

- [ ] **Step 4: Record the exact recovery scope in the final handoff**

```text
Validated commands:
- conda run -n dtgpt python -m unittest tests.test_mimic_prediction_parsing tests.test_prediction_aggregation -v
- conda run -n dtgpt python -m compileall pipeline tests 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row
- conda run -n dtgpt python 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/smoke_check_mimic_local_setup.py

Observed fixes:
- no `ValueError: All arrays must be of the same length`
- no `FutureWarning` from positional `.loc` slicing in `DataFrameConvertTDBDMIMIC.py`
- no `RuntimeWarning: Mean of empty slice` in the mean-merging path
```

- [ ] **Step 5: Commit the verification-only updates if any notes or scripts changed**

```bash
git add pipeline/data_generators/DataFrameConvertTDBDMIMIC.py pipeline/Experiment.py pipeline/prediction_aggregation.py tests/test_mimic_prediction_parsing.py tests/test_prediction_aggregation.py
git commit -m "chore: verify mimic prediction parser recovery"
```
