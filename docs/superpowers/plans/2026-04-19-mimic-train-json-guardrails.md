# MIMIC Train JSON Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the MIMIC BioMistral train/eval pipeline robust to malformed prediction JSON lengths and reduce misleading W&B/asyncio debug noise in Slurm logs.

**Architecture:** Keep the training and evaluation flow intact, but harden the string-to-DataFrame conversion boundary in `DataFrameConvertTDBDMIMIC.py` so malformed model outputs are normalized before Pandas sees them. Then lower default logging verbosity in `Experiment` and make verbose prediction dumps opt-in at the CLI wrapper layer so batch logs show actionable warnings instead of third-party debug chatter.

**Tech Stack:** Python 3.8, pandas, numpy, unittest, Hugging Face/TRL training wrappers, wandb logging, Slurm batch scripts

---

## File Map

- `pipeline/data_generators/DataFrameConvertTDBDMIMIC.py`
  Responsibility: convert BioMistral JSON-like prediction strings into per-patient prediction DataFrames.
- `pipeline/Experiment.py`
  Responsibility: configure experiment logging and handle prediction post-processing fallback during evaluation/merging.
- `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py`
  Responsibility: CLI entrypoint for MIMIC train runs launched by `job/submit_mimic_train_v100.sh`.
- `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py`
  Responsibility: CLI entrypoint for MIMIC eval-only runs.
- `tests/pipeline/test_mimic_prediction_parsing.py`
  Responsibility: regression tests for malformed prediction list lengths and sparse-position expansion.
- `tests/pipeline/test_experiment_logging.py`
  Responsibility: regression tests for default logging level and muted third-party debug loggers.

### Task 1: Harden MIMIC Prediction JSON Parsing

**Files:**
- Create: `tests/pipeline/test_mimic_prediction_parsing.py`
- Modify: `pipeline/data_generators/DataFrameConvertTDBDMIMIC.py:538-636`

- [ ] **Step 1: Write the failing regression tests**

```python
import json
import unittest

import pandas as pd

from pipeline.data_generators.DataFrameConvertTDBDMIMIC import (
    DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC,
)


class ConvertFromStringsToDfTests(unittest.TestCase):
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
                "2026-04-19 00:00:00",
                "2026-04-19 01:00:00",
                "2026-04-19 02:00:00",
            ]
        )
        self.all_column_names = [
            "lab_hr",
            "lab_rr",
            "date",
            "patientid",
            "patient_sample_index",
        ]

    def test_short_column_is_padded_to_prediction_horizon(self):
        prediction = json.dumps(
            {
                "Heart Rate": ["80", "82"],
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
        self.assertEqual(result.loc[0, "lab_hr"], "80")
        self.assertEqual(result.loc[1, "lab_hr"], "82")
        self.assertTrue(pd.isna(result.loc[2, "lab_hr"]))

    def test_long_column_is_truncated_to_prediction_horizon(self):
        prediction = json.dumps(
            {
                "Heart Rate": ["80", "82", "84", "86"],
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
        self.assertEqual(result["lab_hr"].tolist(), ["80", "82", "84"])

    def test_sparse_position_mapping_keeps_declared_hours(self):
        prediction = json.dumps(
            {
                "Heart Rate": ["80", "84"],
                "Respiratory Rate": ["18", "19", "20"],
            }
        )
        prediction_days_column_wise = {
            "Variables to predict for respective hours": {
                "Heart Rate": [1, 3],
                "Respiratory Rate": [1, 2, 3],
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail on the current parser**

Run: `conda run -n dtgpt python -m unittest discover -s tests -p 'test_mimic_prediction_parsing.py' -v`

Expected: FAIL with a traceback ending in `ValueError: All arrays must be of the same length`.

- [ ] **Step 3: Implement normalization before building the DataFrame**

```python
import logging


class DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC:
    @staticmethod
    def _normalize_prediction_list(values, expected_length, column_name):
        if not isinstance(values, list):
            raise ValueError(f"Prediction column {column_name} must decode to a list, got {type(values).__name__}")

        normalized_values = list(values[:expected_length])
        actual_length = len(values)

        if len(normalized_values) < expected_length:
            normalized_values.extend([np.nan] * (expected_length - len(normalized_values)))

        return normalized_values, actual_length

    @staticmethod
    def _normalize_prediction_dictionary(data_as_dic, expected_columns, expected_length):
        normalized = {}
        normalization_issues = []

        for col in expected_columns:
            raw_values = data_as_dic.get(col, [])
            normalized_values, actual_length = (
                DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC._normalize_prediction_list(
                    raw_values,
                    expected_length,
                    col,
                )
            )
            normalized[col] = normalized_values

            if actual_length != expected_length:
                normalization_issues.append((col, actual_length, expected_length))

        return normalized, normalization_issues

    def convert_from_strings_to_df(
        column_name_mapping,
        string_output,
        all_prediction_days,
        patientid,
        patient_sample_index,
        all_column_names,
        all_unk_columns,
        prediction_days_column_wise=None,
    ):
        assert "patientid" in all_column_names and "patient_sample_index" in all_column_names and "date" in all_column_names

        empty_df = pd.DataFrame(columns=all_column_names)
        empty_df.loc[0 : len(all_prediction_days)] = [None] * len(empty_df.columns)
        empty_df["date"] = all_prediction_days
        empty_df["date"] = pd.to_datetime(empty_df["date"])
        empty_df["patientid"] = patientid
        empty_df["patient_sample_index"] = patient_sample_index
        empty_df = empty_df.reset_index()
        empty_df = empty_df.drop(["index"], axis=1, errors="ignore", inplace=False)

        data_as_dic = json.loads(string_output)

        if prediction_days_column_wise is not None:
            column_wise_date_mapping = prediction_days_column_wise["Variables to predict for respective hours"]
            all_positions = sorted({y for x in column_wise_date_mapping.values() for y in x})

            for col in column_wise_date_mapping.keys():
                data_as_dic.setdefault(col, [])
                positions = column_wise_date_mapping[col]
                indices = [all_positions.index(x) for x in positions]

                for idx in range(len(all_positions)):
                    if idx not in indices:
                        data_as_dic[col].insert(idx, np.nan)

            expected_columns = list(column_wise_date_mapping.keys())
            expected_length = len(all_positions)
        else:
            expected_columns = list(data_as_dic.keys())
            expected_length = len(all_prediction_days)

        data_as_dic, normalization_issues = (
            DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC._normalize_prediction_dictionary(
                data_as_dic,
                expected_columns,
                expected_length,
            )
        )

        if normalization_issues:
            logging.warning(
                "Normalized malformed prediction lengths for patient %s sample %s: %s",
                patientid,
                patient_sample_index,
                normalization_issues,
            )

        prediction_df = pd.DataFrame.from_dict(data_as_dic, orient="columns")
        prediction_df.sort_index(inplace=True)

        try:
            new_column_names = (
                DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC._get_columns_short_mapping(
                    prediction_df.columns.to_list(),
                    column_name_mapping,
                )
            )
            prediction_df = prediction_df.rename(columns=new_column_names, inplace=False)
        except Exception:
            print(
                "Making empty dataframe due to column error for patientid: "
                + str(patientid)
                + " and patient_sample_index: "
                + str(patient_sample_index)
            )
            return empty_df

        missing_columns = [col for col in all_column_names if col not in prediction_df.columns.to_list()]
        new_df = pd.DataFrame(index=prediction_df.index, columns=missing_columns)
        prediction_df = pd.concat([prediction_df, new_df], axis=1)

        prediction_df[all_unk_columns] = prediction_df[all_unk_columns].replace("Unknown", "<UNK>", inplace=False)
        prediction_df[all_unk_columns] = prediction_df[all_unk_columns].replace(np.nan, "<UNK>", inplace=False)

        if "progression_progression" in prediction_df.columns:
            prediction_df["progression_progression"] = prediction_df["progression_progression"].replace(
                "Not documented",
                "<UNK>",
                inplace=False,
            )

        diagnosis_columns = [col for col in prediction_df.columns.tolist() if col.startswith("diagnosis_")]
        prediction_df[diagnosis_columns] = prediction_df[diagnosis_columns].where(
            prediction_df[diagnosis_columns] != "diagnosed",
            [x.split("diagnosis.icd10.")[1] for x in prediction_df[diagnosis_columns].columns.to_list()],
            axis=1,
        )

        prediction_df["date"] = all_prediction_days
        prediction_df["date"] = pd.to_datetime(prediction_df["date"])
        prediction_df = prediction_df.loc[:, all_column_names]
        prediction_df["patientid"] = patientid
        prediction_df["patient_sample_index"] = patient_sample_index
        prediction_df = prediction_df.reset_index()
        prediction_df = prediction_df.drop(["index"], axis=1, errors="ignore", inplace=False)
        return prediction_df
```

- [ ] **Step 4: Re-run the parser regression tests**

Run: `conda run -n dtgpt python -m unittest discover -s tests -p 'test_mimic_prediction_parsing.py' -v`

Expected: PASS for all three tests.

- [ ] **Step 5: Commit the parsing guardrails**

```bash
git add tests/pipeline/test_mimic_prediction_parsing.py pipeline/data_generators/DataFrameConvertTDBDMIMIC.py
git commit -m "fix: normalize malformed mimic prediction lengths"
```

### Task 2: Reduce Misleading Log Noise in Train and Eval Runs

**Files:**
- Create: `tests/pipeline/test_experiment_logging.py`
- Modify: `pipeline/Experiment.py:114-129`
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py:20-41`
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py:20-28`

- [ ] **Step 1: Write the failing logging regression test**

```python
import logging
import tempfile
import unittest

from pipeline.Experiment import Experiment


class ExperimentLoggingTests(unittest.TestCase):
    def test_setup_logging_defaults_to_info_and_quiets_noisy_libraries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Experiment(
                "logging_test",
                experiment_folder_root=f"{tmpdir}/",
                timestamp_to_use="2026_04_19___00_00_00_000000",
            )

        self.assertEqual(logging.getLogger().level, logging.INFO)
        self.assertEqual(logging.getLogger("asyncio").level, logging.WARNING)
        self.assertEqual(logging.getLogger("filelock").level, logging.WARNING)
        self.assertEqual(logging.getLogger("urllib3").level, logging.WARNING)
        self.assertEqual(logging.getLogger("git").level, logging.WARNING)
        self.assertEqual(logging.getLogger("wandb").level, logging.WARNING)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the logging test to verify it fails**

Run: `conda run -n dtgpt python -m unittest discover -s tests -p 'test_experiment_logging.py' -v`

Expected: FAIL because the root logger is currently configured at `DEBUG` in `pipeline/Experiment.py`.

- [ ] **Step 3: Lower default verbosity and make prediction dumps opt-in**

```python
# pipeline/Experiment.py
def _setup_logging(self):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.root.handlers = []

    log_file_path = self.get_experiment_folder() + "logfile.log"

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    for logger_name in ("asyncio", "filelock", "urllib3", "git", "wandb"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.info("Logger initiated")
```

```python
# 2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py
def build_parser():
    parser = argparse.ArgumentParser(description="Train DT-GPT on MIMIC-IV with BioMistral.")
    parser.add_argument("--debug", action="store_true", help="Disable WandB logging.")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=10)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=float, default=5)
    parser.add_argument("--eval-interval", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--seq-max-len", type=int, default=3400)
    parser.add_argument("--decimal-precision", type=int, default=1)
    parser.add_argument("--num-samples-to-generate", type=int, default=30)
    parser.add_argument("--sample-merging-strategy", type=str, default="mean")
    parser.add_argument("--max-new-tokens-to-generate", type=int, default=900)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--verbose-predictions",
        action="store_true",
        help="Log raw prompts and decoded prediction strings during evaluation.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    experiment = DTGPT_mimic_biomistral_fft_ti_bd_sr()

    experiment.run(
        debug=args.debug,
        verbose=args.verbose_predictions,
        wandb_prefix_name="DT-GPT - BioMistral - 3.4k - FFT - TI - BD - SR - 30 Samples - Forecast: ",
        wandb_group_name="DT-GPT - BioMistral - FFT - Template Input - Basic Description - Summarized Row",
        train_set="TRAIN",
        validation_set="VALIDATION",
        test_set="TEST",
        learning_rate=args.learning_rate,
        batch_size_training=args.train_batch_size,
        batch_size_validation=args.validation_batch_size,
        weight_decay=0.1,
        gradient_accumulation=args.gradient_accumulation,
        num_train_epochs=args.num_train_epochs,
        eval_interval=args.eval_interval,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler="cosine",
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        nr_days_forecasting=91,
        seq_max_len_in_tokens=args.seq_max_len,
        decimal_precision=args.decimal_precision,
        num_samples_to_generate=args.num_samples_to_generate,
        sample_merging_strategy=args.sample_merging_strategy,
        max_new_tokens_to_generate=args.max_new_tokens_to_generate,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
```

```python
# 2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py
def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a DT-GPT MIMIC-IV checkpoint.")
    parser.add_argument("--eval-model-path", required=True, help="Checkpoint or model directory to evaluate.")
    parser.add_argument("--debug", action="store_true", help="Disable WandB logging.")
    parser.add_argument("--validation-batch-size", type=int, default=5)
    parser.add_argument("--seq-max-len", type=int, default=3400)
    parser.add_argument("--num-samples-to-generate", type=int, default=30)
    parser.add_argument("--max-new-tokens-to-generate", type=int, default=900)
    parser.add_argument(
        "--verbose-predictions",
        action="store_true",
        help="Log raw prompts and decoded prediction strings during evaluation.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    experiment = DTGPT_mimic_biomistral_fft_ti_bd_sr()

    experiment.run(
        debug=args.debug,
        verbose=args.verbose_predictions,
        wandb_prefix_name="DT-GPT - BioMistral - 3.4k - FFT - TI - BD - SR - EVAL 80 - 30 Samples - Forecast: ",
        wandb_group_name="DT-GPT - BioMistral - FFT - Template Input - Basic Description - Summarized Row",
        train_set="TRAIN",
        validation_set="VALIDATION",
        test_set="TEST",
        learning_rate=1e-5,
        batch_size_training=1,
        batch_size_validation=args.validation_batch_size,
        weight_decay=0.1,
        gradient_accumulation=1,
        num_train_epochs=5,
        eval_interval=0.1,
        warmup_ratio=0.10,
        lr_scheduler="cosine",
        nr_days_forecasting=91,
        seq_max_len_in_tokens=args.seq_max_len,
        decimal_precision=1,
        num_samples_to_generate=args.num_samples_to_generate,
        sample_merging_strategy="mean",
        max_new_tokens_to_generate=args.max_new_tokens_to_generate,
        eval_model_path=args.eval_model_path,
    )
```

- [ ] **Step 4: Re-run unit tests and a syntax check**

Run: `conda run -n dtgpt python -m unittest discover -s tests -p 'test_experiment_logging.py' -v`

Expected: PASS.

Run: `conda run -n dtgpt python -m compileall pipeline 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row`

Expected: PASS with no syntax errors.

- [ ] **Step 5: Commit the logging changes**

```bash
git add tests/pipeline/test_experiment_logging.py \
    pipeline/Experiment.py \
    1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py \
    1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py
git commit -m "fix: reduce mimic training log noise"
```

### Task 3: End-to-End Verification Against the Original Failure Mode

**Files:**
- Reference: `job/submit_mimic_train_v100.sh`
- Test: `logs/mimic_train_<jobid>.out`

- [ ] **Step 1: Run focused local verification before re-submitting Slurm**

```bash
conda run -n dtgpt python -m unittest discover -s tests -p 'test_mimic_prediction_parsing.py' -v
conda run -n dtgpt python -m unittest discover -s tests -p 'test_experiment_logging.py' -v
conda run -n dtgpt python -m compileall pipeline 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row
```

Expected: PASS for both unittest suites and no syntax errors from `compileall`.

- [ ] **Step 2: Re-submit the MIMIC train job with default logging settings**

Run: `sbatch job/submit_mimic_train_v100.sh`

Expected: Slurm prints a new job ID.

- [ ] **Step 3: Verify the new Slurm log shows normalized warnings instead of Pandas length crashes**

Run: `rg -n "Normalized malformed prediction lengths|All arrays must be of the same length|socket.send\\(\\) raised exception|asyncio - DEBUG|filelock - DEBUG" logs/mimic_train_<new_jobid>.out`

Expected: zero matches for `All arrays must be of the same length`, zero matches for `asyncio - DEBUG`, zero matches for `filelock - DEBUG`, and only bounded `Normalized malformed prediction lengths` warnings when the model emits malformed arrays.

- [ ] **Step 4: Commit any follow-up verification note if documentation is needed**

```bash
git status --short
```

Expected: no unexpected tracked-file changes beyond the two implementation commits above. If a short verification note is added later, use a separate docs-only commit.
