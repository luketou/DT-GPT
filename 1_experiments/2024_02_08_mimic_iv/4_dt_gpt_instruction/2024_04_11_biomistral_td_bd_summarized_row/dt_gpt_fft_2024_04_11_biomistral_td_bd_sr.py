import __init__
import hashlib
import os
import sys
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Unsloth must patch transformers/TRL/PEFT before those packages are imported.
# It also returns empty logits by default, while this TRL SFTTrainer version
# computes entropy from outputs.logits after loss computation.
if os.environ.get("DTGPT_USE_UNSLOTH") == "1":
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
    import unsloth  # noqa: F401

from pipeline.EvaluationManager import EvaluationManager
from pipeline.Experiment import Experiment
import wandb
import pandas as pd
import logging
from pipeline.data_generators.DataFrameConvertTDBDMIMIC import DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC
from pipeline.DFConversionHelpers import iter_converted_tuples, log_memory_usage, process_all_tuples_multiprocessing
from pipeline.data_processors.DataProcessorBiomistral import DataProcessorBiomistral
from pipeline.NormalizationFilterManager import Only_Double3_sigma_Filtering
import torch
from transformers import AutoModelForCausalLM, TrainerCallback
from peft import PeftModel
from pipeline.hf_training_args import create_training_arguments
import gc
from pipeline.NormalizationFilterManager import Only_Double3_sigma_Filtering
from pipeline.MetricManager import MetricManager
from trl import SFTTrainer
import json
import inspect
from pipeline.Splitters import After24HSplitter
from pipeline.NormalizationFilterManager import Only_Standardization
from pipeline.local_paths import (
    get_biomistral_model_path,
    get_mimic_column_descriptive_mapping_path,
    get_mimic_constants_path,
    get_mimic_dataset_statistics_path,
    get_model_load_kwargs,
    get_precision_config,
    get_torch_dtype,
)
from pipeline.lora_helpers import (
    apply_lora_to_model,
    build_lora_adapter_path,
    build_mistral_lora_config,
    load_lora_model_for_inference,
)
from pipeline.evaluation_shards import shard_suffix, slice_by_shard
from pipeline.distributed_dataset_cache import (
    MANIFEST_FILE,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    build_dataset_cache_dir,
    dataset_cache_complete,
    dataset_cache_paths,
    dataset_cache_temp_dir,
    mark_dataset_cache_complete,
    wait_for_dataset_cache_complete,
)
from datasets import concatenate_datasets, load_from_disk


class PreserveEpochCheckpointCallback(TrainerCallback):
    """Copy selected epoch-end checkpoints outside HF checkpoint rotation."""

    def __init__(self, epochs_to_preserve):
        self.epochs_to_preserve = {int(epoch) for epoch in epochs_to_preserve if int(epoch) >= 1}
        self._preserved_epochs = set()

    def on_save(self, args, state, control, **kwargs):
        if not self.epochs_to_preserve or not getattr(state, "is_world_process_zero", True):
            return control
        if state.epoch is None:
            return control

        epoch_value = float(state.epoch)
        rounded_epoch = round(epoch_value)
        if abs(epoch_value - rounded_epoch) > 1e-3:
            return control
        if rounded_epoch not in self.epochs_to_preserve or rounded_epoch in self._preserved_epochs:
            return control

        source_checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not source_checkpoint_dir.exists():
            logging.warning(
                "Epoch %s preservation skipped because checkpoint directory does not exist: %s",
                rounded_epoch,
                source_checkpoint_dir,
            )
            return control

        preserved_root = Path(args.output_dir) / "preserved_epoch_checkpoints"
        preserved_root.mkdir(parents=True, exist_ok=True)
        destination_dir = preserved_root / f"epoch-{rounded_epoch}-step-{state.global_step}"
        if destination_dir.exists():
            self._preserved_epochs.add(rounded_epoch)
            return control

        shutil.copytree(source_checkpoint_dir, destination_dir)
        logging.info(
            "Preserved epoch %s checkpoint outside HF rotation: %s -> %s",
            rounded_epoch,
            source_checkpoint_dir,
            destination_dir,
        )
        self._preserved_epochs.add(rounded_epoch)
        return control


class DTGPT_mimic_biomistral_fft_ti_bd_sr:

    def run(self, debug=False, verbose=True,
                wandb_prefix_name="DT-GPT - Meditron FFT - Completion Loss - Forecast: ",
                wandb_group_name="DT-GPT - Meditron FFT - Completion",
                train_set= "TRAIN", validation_set = "VALIDATION", test_set = "TEST",
                learning_rate=1e-5, batch_size_training=1, batch_size_validation=1, weight_decay=0.1, gradient_accumulation=1,
                num_train_epochs=1.0, eval_interval=0.25, warmup_ratio=0.1, lr_scheduler="cosine", gradient_checkpointing=False,
                logging_steps=10,
                max_steps=-1,
                resume_from_checkpoint=None,
                preserve_epoch_checkpoints=None,
                nr_days_forecasting=91, seq_max_len_in_tokens=4000, decimal_precision=1,
                gen_num_beams=1, gen_do_sample=False,
                eval_model_path=None,
                num_samples_to_generate=10, sample_merging_strategy="mean",
                max_new_tokens_to_generate=1200,
                use_lora=True,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                use_dora=False,
                use_unsloth=False,
                deepspeed_config=None,
                sft_dataset_num_proc=1,
                df_conversion_n_jobs=None,
                train_max_patients=None,
                validation_max_patients=None,
                test_max_patients=None,
                train_max_samples=None,
                validation_max_samples=None,
                dataset_cache_mode="auto",
                dataset_cache_name=None,
                dataset_cache_build_chunk_size=256,
                skip_eval=False,
                eval_backend="vllm",
                eval_shard_index=0,
                eval_num_shards=1,
                eval_max_samples=None,
                prediction_url="http://127.0.0.1:18101/v1/",
                vllm_model_name=None,
                max_concurrent_requests=16,
                vllm_temperature=1.0,
                vllm_top_p=0.9,
                vllm_total_max_length=4092,
                vllm_dynamic_max_tokens=True,
                vllm_minimum_max_tokens=1,
                vllm_fail_on_request_error=True):


        ######################################################## SETUP EXPERIMENT ########################################################

        MIN_NR_DAYS_FORECAST = 24   # We want to forecast up to the first visit after this value, or until the start of the next therapy (which ever comes first) - using 91 since it is the closest multiple of 7 to 90 days - often used for meds
        SEQUENCE_MAX_LENGTH_IN_TOKENS = seq_max_len_in_tokens
        MODEL_HF_NAME = get_biomistral_model_path()
        DECIMAL_PRECISION = decimal_precision
        precision_config = get_precision_config(training=True)
        rank = int(os.environ.get("RANK", "0"))
        is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        is_main_process = rank == 0
        if dataset_cache_mode not in {"auto", "build-only", "require"}:
            raise ValueError("dataset_cache_mode must be one of: auto, build-only, require")
        if dataset_cache_build_chunk_size <= 0:
            raise ValueError("dataset_cache_build_chunk_size must be positive")
        preserve_epoch_checkpoints = sorted(
            {int(epoch) for epoch in (preserve_epoch_checkpoints or []) if int(epoch) >= 1}
        )

        def checkpoint_has_deepspeed_state(checkpoint_path):
            if checkpoint_path is None:
                return False
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                return False
            if (checkpoint_path / "latest").exists():
                return True
            return any(child.is_dir() and child.name.startswith("global_step") for child in checkpoint_path.iterdir())

        def read_checkpoint_global_step(checkpoint_path):
            trainer_state_path = Path(checkpoint_path) / "trainer_state.json"
            if not trainer_state_path.exists():
                return 0
            with open(trainer_state_path) as handle:
                trainer_state = json.load(handle)
            return int(trainer_state.get("global_step") or 0)

        trainer_resume_checkpoint = resume_from_checkpoint
        adapter_init_checkpoint = None
        checkpoint_global_step = 0
        effective_max_steps = max_steps
        if resume_from_checkpoint is not None:
            resume_checkpoint_path = Path(resume_from_checkpoint)
            resume_checkpoint_has_deepspeed_state = checkpoint_has_deepspeed_state(resume_checkpoint_path)
            should_initialize_from_adapter = (
                use_unsloth
                or (deepspeed_config is not None and not resume_checkpoint_has_deepspeed_state)
            )
            if should_initialize_from_adapter:
                adapter_path = resume_checkpoint_path / "adapter_model.safetensors"
                if not adapter_path.exists():
                    raise ValueError(
                        "Resume checkpoint must contain adapter_model.safetensors for "
                        "adapter-initialized training, but it was not found: "
                        f"adapter_model.safetensors: {resume_checkpoint_path}"
                    )
                checkpoint_global_step = read_checkpoint_global_step(resume_checkpoint_path)
                adapter_init_checkpoint = str(resume_checkpoint_path)
                trainer_resume_checkpoint = None
                if max_steps is not None and max_steps > 0:
                    effective_max_steps = max(max_steps - checkpoint_global_step, 1)
                logging.info(
                    "Checkpoint %s will initialize trainable adapter weights instead of "
                    "resuming full Trainer/DeepSpeed state; "
                    "initializing adapter weights and training for remaining steps. "
                    "checkpoint_global_step=%s target_max_steps=%s effective_max_steps=%s "
                    "use_unsloth=%s deepspeed=%s checkpoint_has_deepspeed_state=%s",
                    resume_checkpoint_path,
                    checkpoint_global_step,
                    max_steps,
                    effective_max_steps,
                    use_unsloth,
                    deepspeed_config is not None,
                    resume_checkpoint_has_deepspeed_state,
                )
                if deepspeed_config is not None and gradient_checkpointing:
                    logging.info(
                        "Disabling gradient checkpointing for PEFT-adapter DeepSpeed resume fallback "
                        "because torch checkpoint recomputation can produce mismatched tensor metadata "
                        "with this stack."
                    )
                    gradient_checkpointing = False


        # Setup hyperparameters
        LEARNING_RATE = learning_rate                # From meditron paper (pretraining setting)
        BATCH_SIZE_TRAINING = batch_size_training             # For debugging purposes
        BATCH_SIZE_VALIDATION = batch_size_validation
        WEIGHT_DECAY = weight_decay                  # From meditron paper
        GRADIENT_ACCUMULATION = gradient_accumulation

        NUM_TRAIN_EPOCHS = num_train_epochs              # 10% of training data
        EVAL_NUM_STEPS = eval_interval               # So 4 times in one training run

        WARMUP_RATIO = warmup_ratio                  # From meditron paper
        LR_SCHEDULER_TYPE = lr_scheduler        # From meditron paper


        eval_manager = EvaluationManager("2024_03_15_mimic_iv")
        if eval_backend not in ["hf", "vllm"]:
            raise ValueError("eval_backend must be either 'hf' or 'vllm'")

        experiment = Experiment(
            "setup",
            timestamp_to_use=os.environ.get("DTGPT_RUN_TIMESTAMP"),
        )

        # Uncomment for debug mode of WandB
        if debug:
            experiment.setup_wandb_debug_mode()
        elif is_main_process:
            experiment.setup_wandb(wandb_prefix_name + str(MIN_NR_DAYS_FORECAST) + " LR: " + str(learning_rate), wandb_group_name, project="UC - MIMIC-IV")
            wandb.config.update({"generation_config": { "gen_num_beams": gen_num_beams,
                                    "gen_do_sample": gen_do_sample}},
                                    allow_val_change=True)


        ############################################################ Load & Split Data ############################################################


        training_full_paths, training_full_patientids = eval_manager.get_paths_to_events_in_split(train_set)
        validation_full_paths, validation_full_patientids = eval_manager.get_paths_to_events_in_split(validation_set)
        test_full_paths, test_full_patientids = eval_manager.get_paths_to_events_in_split(test_set)

        def limit_split(paths, patientids, max_patients, split_name):
            if max_patients is None:
                return paths, patientids
            if max_patients <= 0:
                raise ValueError(f"{split_name} max patients must be positive, got {max_patients}")
            limited_count = min(max_patients, len(patientids))
            logging.info(
                "Limiting %s patient IDs from %s to %s using explicit smoke limit",
                split_name,
                len(patientids),
                limited_count,
            )
            return paths[:limited_count], patientids[:limited_count]

        training_full_paths, training_full_patientids = limit_split(
            training_full_paths, training_full_patientids, train_max_patients, "train"
        )
        validation_full_paths, validation_full_patientids = limit_split(
            validation_full_paths, validation_full_patientids, validation_max_patients, "validation"
        )
        test_full_paths, test_full_patientids = limit_split(
            test_full_paths, test_full_patientids, test_max_patients, "test"
        )

        target_cols = eval_manager.get_column_usage()[2]
        split_fraction = os.environ.get("DTGPT_PATIENT_SPLIT_FRACTION", "default")
        dataset_cache_manifest = {
            "cache_format_version": 2,
            "train_set": train_set,
            "validation_set": validation_set,
            "patient_split_fraction": split_fraction,
            "seq_max_len": SEQUENCE_MAX_LENGTH_IN_TOKENS,
            "decimal_precision": DECIMAL_PRECISION,
            "tokenizer_path": MODEL_HF_NAME,
            "target_variables": target_cols,
            "normalization_statistics_path": str(get_mimic_dataset_statistics_path()),
            "train_max_patients": train_max_patients,
            "validation_max_patients": validation_max_patients,
            "train_max_samples": train_max_samples,
            "validation_max_samples": validation_max_samples,
            "df_conversion_n_jobs": df_conversion_n_jobs,
            "sft_dataset_num_proc": sft_dataset_num_proc,
            "code_version": os.environ.get("DTGPT_CODE_VERSION", "unknown"),
        }
        manifest_json = json.dumps(dataset_cache_manifest, sort_keys=True, separators=(",", ":"))
        manifest_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()[:12]
        if dataset_cache_name is None:
            try:
                split_name = str(int(round(float(split_fraction) * 100)))
            except ValueError:
                split_name = str(split_fraction).replace(".", "p")
            dataset_cache_name = (
                f"mimic_tokenized_seq{SEQUENCE_MAX_LENGTH_IN_TOKENS}"
                f"_split{split_name}_dp{DECIMAL_PRECISION}_{manifest_hash}"
            )
        dataset_cache_manifest["manifest_hash"] = manifest_hash
        dataset_cache_manifest["cache_name"] = dataset_cache_name
        dataset_cache_root = os.environ.get("DTGPT_DATASET_CACHE_ROOT")
        if dataset_cache_root:
            dataset_cache_dir = Path(dataset_cache_root) / dataset_cache_name
        else:
            dataset_cache_dir = build_dataset_cache_dir(
                experiment.get_experiment_folder(),
                dataset_cache_name,
            )
        dataset_paths = dataset_cache_paths(dataset_cache_dir)
        dataset_cache_is_complete = dataset_cache_complete(dataset_cache_dir)
        dataset_cache_manifest_matches = False
        if dataset_cache_is_complete:
            cached_manifest_path = dataset_cache_dir / MANIFEST_FILE
            with open(cached_manifest_path, encoding="utf-8") as handle:
                cached_manifest = json.load(handle)
            dataset_cache_manifest_matches = cached_manifest.get("manifest_hash") == manifest_hash
            if not dataset_cache_manifest_matches:
                logging.warning(
                    "Tokenized dataset cache manifest mismatch at %s: expected hash %s, found %s",
                    cached_manifest_path,
                    manifest_hash,
                    cached_manifest.get("manifest_hash"),
                )
                if dataset_cache_mode in {"auto", "build-only"}:
                    dataset_cache_is_complete = False
        if is_main_process:
            logging.info(
                "Dataset cache mode=%s name=%s path=%s complete=%s manifest_matches=%s",
                dataset_cache_mode,
                dataset_cache_name,
                dataset_cache_dir,
                dataset_cache_is_complete,
                dataset_cache_manifest_matches,
            )

        if eval_model_path is None and dataset_cache_mode == "require" and not dataset_cache_is_complete:
            raise RuntimeError(
                "Required tokenized dataset cache is missing or incomplete: "
                f"{dataset_cache_dir}. Build it first with "
                "job/submit_mimic_build_tokenized_cache_cpu.sh "
                "or rerun with --dataset-cache-mode build-only."
            )
        if eval_model_path is None and dataset_cache_mode == "require" and not dataset_cache_manifest_matches:
            raise RuntimeError(
                "Required tokenized dataset cache manifest does not match current settings: "
                f"{dataset_cache_dir / MANIFEST_FILE}. Build a matching cache first with "
                "job/submit_mimic_build_tokenized_cache_cpu.sh."
            )

        should_load_training_validation_dfs = (
            eval_model_path is None
            and dataset_cache_mode != "require"
            and not dataset_cache_is_complete
        )
        should_load_test_dfs = not skip_eval

        training_full_constants = []
        training_full_events = []
        validation_full_constants = []
        validation_full_events = []
        test_full_constants = []
        test_full_events = []

        if should_load_training_validation_dfs:
            training_full_constants, training_full_events = eval_manager.load_list_of_patient_dfs_and_constants(training_full_patientids)
            validation_full_constants, validation_full_events = eval_manager.load_list_of_patient_dfs_and_constants(validation_full_patientids)
        else:
            logging.info("Skipping train/validation raw DF load because tokenized cache will be used.")

        if should_load_test_dfs:
            test_full_constants, test_full_events = eval_manager.load_list_of_patient_dfs_and_constants(test_full_patientids)
        else:
            logging.info("Skipping test raw DF load because skip_eval=True.")
        log_memory_usage("after-conditional-raw-df-load")

        # Setup splitter object
        splitter = After24HSplitter()

        def limit_events(events, meta_data, max_samples, split_name):
            if max_samples is None:
                return events, meta_data
            if max_samples <= 0:
                raise ValueError(f"{split_name} max samples must be positive, got {max_samples}")
            limited_count = min(max_samples, len(events))
            logging.info(
                "Limiting %s split samples from %s to %s using explicit smoke limit",
                split_name,
                len(events),
                limited_count,
            )
            return events[:limited_count], meta_data[:limited_count]

        training_events = []
        training_meta_data = []
        if should_load_training_validation_dfs:
            training_events, training_meta_data = splitter.setup_split_indices(training_full_events, eval_manager)
            training_events, training_meta_data = limit_events(
                training_events, training_meta_data, train_max_samples, "train"
            )

        # Setup also validation and test
        validation_events = []
        validation_meta = []
        if should_load_training_validation_dfs:
            validation_events, validation_meta = splitter.setup_split_indices(validation_full_events, eval_manager)
            validation_events, validation_meta = limit_events(
                validation_events, validation_meta, validation_max_samples, "validation"
            )

        test_events = []
        test_meta = []
        test_events_full_count = 0
        if should_load_test_dfs:
            test_events, test_meta = splitter.setup_split_indices(test_full_events, eval_manager)
            test_events_full_count = len(test_events)
            test_events, test_meta = slice_by_shard(
                test_events,
                test_meta,
                shard_index=eval_shard_index,
                num_shards=eval_num_shards,
            )
            if eval_max_samples is not None:
                test_events = test_events[:eval_max_samples]
                test_meta = test_meta[:eval_max_samples]
        else:
            test_events_full_count = 0
        test_set_output_name = test_set + shard_suffix(eval_shard_index, eval_num_shards)
        logging.info(
            "Using eval backend %s; test shard %s/%s contains %s of %s samples; eval_max_samples=%s",
            eval_backend,
            eval_shard_index,
            eval_num_shards,
            len(test_events),
            test_events_full_count,
            eval_max_samples,
        )
        del training_full_paths, validation_full_paths, test_full_paths
        del test_full_constants
        gc.collect()
        log_memory_usage("after-splitting-and-dropping-unused-paths")

        path_to_statistics_file = str(get_mimic_dataset_statistics_path())
        with open(path_to_statistics_file) as f:
            statistics_dic = json.load(f)


        ######################################################## CONVERT DFS TO STRINGS ########################################################

        logging.info("Converting DFs to Strings")

        def filtering_rows_rest_budget(df, nr_tokens_budget):


            #: create summarizing row for every variable for its respective last observed value, and put into it first row

            # Forward fill the missing values
            df_filled = df.ffill()
            # Take the first row (which corresponds to the last non-NaN observation in the original DataFrame)
            last_valid_values = df_filled.iloc[-1]
            # Assign these values to the first row of the original DataFrame
            df.iloc[0] = last_valid_values
            # If you want to maintain the original index, you can do the following:
            df.loc[df.index[0]] = last_valid_values
            # set to nan if these values are seen in last row (this is just a simple heuristic to remove some tokens)
            first_row_label = df.index[0]
            original_basics = df.loc[first_row_label, ["date", "patientid", "patient_sample_index"]].copy()
            mask = df.iloc[0] == df.iloc[-1]
            df.loc[first_row_label, mask] = pd.NA

            #: adjust budgets of row --> 6 tokens per non na value
            budget_of_first_row = 6 * df.iloc[0].notna().sum()
            df.loc[first_row_label, "estimated_nr_tokens"] = budget_of_first_row

            # Add back in basics
            df.loc[first_row_label, ["date", "patientid", "patient_sample_index"]] = original_basics
            df.loc[first_row_label, "date"] = -1  # So that model gets it quicker


            # Add in buffer to ensure we don't go over the budget
            buffer = 300
            nr_tokens_budget = max(nr_tokens_budget - buffer, 0)

            df_reversed = df.iloc[::-1].copy()
            # Calculate the cumulative sum of the estimated_nr_tokens column
            df_reversed['cumulative_tokens'] = df_reversed['estimated_nr_tokens'].cumsum()

            #: ensure summarizing row is kept in
            df_reversed.loc[first_row_label, "cumulative_tokens"] = 0

            # Filter rows where the cumulative sum is less than or equal to nr_tokens_budget
            df_filtered = df_reversed[df_reversed['cumulative_tokens'] <= nr_tokens_budget]
            # Reverse the DataFrame back to the original order
            df_result = df_filtered.iloc[::-1]
            # drop cumulative tokens column
            df_result = df_result.drop(columns=['cumulative_tokens'])
            return df_result



        # Setup column mapping
        column_mapping = pd.read_csv(get_mimic_column_descriptive_mapping_path())

        # Setup statistics paths
        path_to_statistics_file = str(get_mimic_dataset_statistics_path())
        with open(path_to_statistics_file) as f:
            statistics_dic = json.load(f)

        # Setup conversion function
        prompt = "Please predict for the previously noted down hours the lab variables for this intensive care unit patient, in a JSON format as in the patient history, with the same ordering of variables as in the 'Variables to predict' section."


        # Setup constant column mapping {<original_colunn_name>: <descriptive_column_name>}
        constant_column_mapping = {}

        # Get random constant row
        if training_full_constants:
            constant_row = training_full_constants[0]
        elif test_full_constants:
            constant_row = test_full_constants[0]
        else:
            constant_row = pd.read_csv(get_mimic_constants_path(), nrows=1)
        constant_columns = constant_row.columns.tolist()
        # Get mapping for indications
        for col in constant_columns:
            match = column_mapping[column_mapping["original_column_names"] == col]
            if match.shape[0] > 0:
                constant_column_mapping[col] = match["descriptive_column_name"].values[0]

        # add basics
        constant_column_mapping["Age"] = "age"
        constant_column_mapping["gender"] = "gender"
        constant_column_mapping["ethnicity"] = "ethnicity"
        constant_column_mapping["insurance"] = "insurance"


        conversion_function = DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC.convert_df_to_strings

        if eval_model_path is None:
            should_build_dataset_cache = (
                is_main_process
                and dataset_cache_mode in {"auto", "build-only"}
                and not dataset_cache_is_complete
            )

            if is_main_process and dataset_cache_is_complete:
                logging.info("Using existing tokenized dataset cache at %s", dataset_cache_dir)
                if training_full_constants or training_full_events:
                    del training_full_constants, training_full_events
                if validation_full_constants or validation_full_events:
                    del validation_full_constants, validation_full_events
                if training_events or validation_events:
                    del training_events, validation_events
                gc.collect()
                log_memory_usage("rank-main-after-dropping-raw-dfs-for-existing-cache")

            if is_distributed and not is_main_process:
                logging.info(
                    "Rank %s skipping DF conversion and waiting for rank 0 dataset cache at %s",
                    rank,
                    dataset_cache_dir,
                )
                if training_full_constants or training_full_events:
                    del training_full_constants, training_full_events
                if validation_full_constants or validation_full_events:
                    del validation_full_constants, validation_full_events
                if training_events or validation_events:
                    del training_events, validation_events
                gc.collect()
                log_memory_usage("rank-non-main-after-dropping-raw-dfs-before-cache-wait")


        ######################################################## SETUP DATA PROCESSOR ########################################################


        # Load data processor
        logging.info("Setting up data processor")

        dp = DataProcessorBiomistral(experiment, path_to_statistics_file, column_mapping,
                                    model_to_use=MODEL_HF_NAME, max_total_length=SEQUENCE_MAX_LENGTH_IN_TOKENS,
                                    collator_setting="completion")
        dp.set_converter(DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC)
        dp.set_for_training()

        dp.setup_cols(target_cols)

        if eval_model_path is None:
            tokenize = True

            def normalize_event_for_tokenization(event_tuple, normalization_filter):
                curr_const_row, true_events_input, true_future_events_input, target_dataframe = event_tuple
                normalized_input = normalization_filter.normalize_and_filter(
                    true_events_input.copy(),
                    None,
                    replace_nan_rows=False,
                    replace_missing_in_prediction=False,
                    verbose=False,
                    specific_column_list=["lab_26499_4"],
                )[0]
                normalized_target = normalization_filter.normalize_and_filter(
                    target_dataframe.copy(),
                    None,
                    replace_nan_rows=False,
                    replace_missing_in_prediction=False,
                    verbose=False,
                )[0]
                return (
                    column_mapping,
                    curr_const_row,
                    normalized_input,
                    true_future_events_input,
                    normalized_target,
                    filtering_rows_rest_budget,
                    SEQUENCE_MAX_LENGTH_IN_TOKENS,
                    DECIMAL_PRECISION,
                    prompt,
                    constant_column_mapping,
                )

            def save_tokenized_split_cache(raw_events, split_name, output_path, temp_cache_dir, normalization_filter):
                shard_root = temp_cache_dir / f"{split_name}_shards"
                if shard_root.exists():
                    shutil.rmtree(shard_root)
                shard_root.mkdir(parents=True, exist_ok=True)
                shard_paths = []
                total_events = len(raw_events)
                logging.info(
                    "Building %s tokenized cache in chunks: samples=%s chunk_size=%s",
                    split_name,
                    total_events,
                    dataset_cache_build_chunk_size,
                )
                if total_events == 0:
                    raise RuntimeError(f"No {split_name} events are available for dataset cache build.")

                for chunk_start in range(0, total_events, dataset_cache_build_chunk_size):
                    chunk_end = min(chunk_start + dataset_cache_build_chunk_size, total_events)
                    chunk_index = len(shard_paths)
                    logging.info(
                        "Building %s tokenized cache chunk %s covering samples [%s, %s)",
                        split_name,
                        chunk_index,
                        chunk_start,
                        chunk_end,
                    )
                    conversion_args = [
                        normalize_event_for_tokenization(event_tuple, normalization_filter)
                        for event_tuple in raw_events[chunk_start:chunk_end]
                    ]
                    records = list(iter_converted_tuples(conversion_args, conversion_function, log_every=100))
                    if split_name == TRAIN_SPLIT and chunk_index == 0 and len(records) >= 2:
                        logging.info("Example of input: " + records[0]["input_text"])
                        logging.info("Example of target: " + records[0]["target_text"])
                        logging.info("Example of input: " + records[1]["input_text"])
                        logging.info("Example of target: " + records[1]["target_text"])

                    tokenized_chunk = dp.preprocess_converted_records(
                        records,
                        tokenize=tokenize,
                        keep_text_columns=False,
                    )
                    shard_path = shard_root / f"shard_{chunk_index:05d}"
                    tokenized_chunk.save_to_disk(str(shard_path))
                    shard_paths.append(shard_path)
                    del conversion_args, records, tokenized_chunk
                    gc.collect()
                    log_memory_usage(f"after-saving-{split_name}-tokenized-cache-chunk-{chunk_index}")

                logging.info("Combining %s %s tokenized cache shards", len(shard_paths), split_name)
                if len(shard_paths) == 1:
                    combined_dataset = load_from_disk(str(shard_paths[0]))
                else:
                    combined_dataset = concatenate_datasets([
                        load_from_disk(str(shard_path)) for shard_path in shard_paths
                    ])
                combined_dataset.save_to_disk(str(output_path))
                del combined_dataset
                gc.collect()
                shutil.rmtree(shard_root)
                log_memory_usage(f"after-saving-final-{split_name}-tokenized-cache")

            if should_build_dataset_cache:
                logging.info("Rank 0 building tokenized dataset cache at %s", dataset_cache_dir)
                temp_cache_dir = dataset_cache_temp_dir(dataset_cache_dir)
                if temp_cache_dir.exists():
                    shutil.rmtree(temp_cache_dir)
                temp_cache_dir.mkdir(parents=True, exist_ok=True)
                temp_dataset_paths = dataset_cache_paths(temp_cache_dir)

                training_norm_filter = Only_Double3_sigma_Filtering(path_to_statistics_file)
                save_tokenized_split_cache(
                    training_events,
                    TRAIN_SPLIT,
                    temp_dataset_paths[TRAIN_SPLIT],
                    temp_cache_dir,
                    training_norm_filter,
                )
                del training_events
                if training_full_constants or training_full_events:
                    del training_full_constants, training_full_events
                gc.collect()
                log_memory_usage("after-training-tokenized-cache-build-and-raw-train-free")

                save_tokenized_split_cache(
                    validation_events,
                    VALIDATION_SPLIT,
                    temp_dataset_paths[VALIDATION_SPLIT],
                    temp_cache_dir,
                    training_norm_filter,
                )
                del validation_events
                if validation_full_constants or validation_full_events:
                    del validation_full_constants, validation_full_events
                gc.collect()
                log_memory_usage("after-validation-tokenized-cache-build-and-raw-validation-free")

                manifest_path = temp_cache_dir / MANIFEST_FILE
                manifest_path.write_text(
                    json.dumps(dataset_cache_manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                mark_dataset_cache_complete(temp_cache_dir)
                old_cache_dir = dataset_cache_dir.with_name(dataset_cache_dir.name + ".old")
                if old_cache_dir.exists():
                    shutil.rmtree(old_cache_dir)
                if dataset_cache_dir.exists():
                    dataset_cache_dir.rename(old_cache_dir)
                try:
                    temp_cache_dir.rename(dataset_cache_dir)
                except Exception:
                    if old_cache_dir.exists() and not dataset_cache_dir.exists():
                        old_cache_dir.rename(dataset_cache_dir)
                    raise
                if old_cache_dir.exists():
                    shutil.rmtree(old_cache_dir)
                logging.info("Saved tokenized dataset cache at %s", dataset_cache_dir)

                gc.collect()
                log_memory_usage("after-saving-tokenized-dataset-cache")

            if is_distributed and not is_main_process:
                wait_for_dataset_cache_complete(dataset_cache_dir, poll_seconds=30)

            if not dataset_cache_complete(dataset_cache_dir):
                raise RuntimeError(f"Tokenized dataset cache was not created: {dataset_cache_dir}")

            if dataset_cache_mode == "build-only":
                logging.info("Dataset cache build-only mode complete; exiting before model load/training.")
                if is_main_process and wandb.run is not None:
                    wandb.run.finish()
                return

            logging.info("Loading tokenized dataset cache from %s", dataset_cache_dir)
            training_dataset = load_from_disk(str(dataset_paths[TRAIN_SPLIT]))
            validation_dataset = load_from_disk(str(dataset_paths[VALIDATION_SPLIT]))
            gc.collect()
            log_memory_usage("after-loading-tokenized-dataset-cache")


            ######################################################## SETUP MODEL ########################################################

            logging.info("Setting up model")

            if use_unsloth:
                from pipeline.unsloth_helpers import load_model_unsloth, apply_unsloth_peft
                model, _unsloth_tokenizer = load_model_unsloth(
                    MODEL_HF_NAME,
                    max_seq_length=SEQUENCE_MAX_LENGTH_IN_TOKENS,
                    load_in_4bit=True,
                    dtype=get_torch_dtype(precision_config["torch_dtype_name"]),
                )
                if adapter_init_checkpoint is not None:
                    logging.info("Loading trainable PEFT adapter into Unsloth model from checkpoint: %s", adapter_init_checkpoint)
                    model = PeftModel.from_pretrained(
                        model,
                        adapter_init_checkpoint,
                        is_trainable=True,
                    )
                    model.print_trainable_parameters()
                else:
                    model = apply_unsloth_peft(
                        model,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        use_dora=use_dora,
                        use_gradient_checkpointing=gradient_checkpointing,
                    )
            else:
                model_load_kwargs = get_model_load_kwargs(
                    experiment.model_cache_path,
                    training=True,
                )

                # Load in Meditron
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_HF_NAME,
                    **model_load_kwargs,
                )

                if gradient_checkpointing:
                    model.config.use_cache = False

                if use_lora:
                    if adapter_init_checkpoint is not None:
                        if gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
                            model.enable_input_require_grads()
                        logging.info("Loading trainable PEFT adapter from checkpoint: %s", adapter_init_checkpoint)
                        model = PeftModel.from_pretrained(
                            model,
                            adapter_init_checkpoint,
                            is_trainable=True,
                        )
                        model.print_trainable_parameters()
                    else:
                        lora_config = build_mistral_lora_config(
                            r=lora_r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                            use_dora=use_dora,
                        )
                        model = apply_lora_to_model(
                            model,
                            lora_config,
                            gradient_checkpointing=gradient_checkpointing,
                        )

            logging.info("Num params in model: " + str(model.num_parameters()))

            # Load data collator
            data_collator = dp.get_collator(model)


            train_params = create_training_arguments(
                output_dir=experiment.model_path,
                per_device_train_batch_size=BATCH_SIZE_TRAINING,
                per_device_eval_batch_size=BATCH_SIZE_VALIDATION,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION,
                gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
                optim="adamw_torch",
                evaluation_strategy="steps",
                save_strategy="steps",
                save_steps=EVAL_NUM_STEPS,
                eval_steps=EVAL_NUM_STEPS,
                logging_steps=logging_steps,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                fp16=precision_config["fp16"],
                bf16=precision_config["bf16"],
                deepspeed=deepspeed_config,
                ddp_find_unused_parameters=False if is_distributed and use_unsloth else None,
                num_train_epochs=NUM_TRAIN_EPOCHS,
                warmup_ratio=WARMUP_RATIO,
                group_by_length=True,
                lr_scheduler_type=LR_SCHEDULER_TYPE,
                lr_scheduler_kwargs={},
                push_to_hub=False,
                save_total_limit=2,
                report_to="wandb" if is_main_process else "none",
                load_best_model_at_end=True,
                max_steps=effective_max_steps,
                seed=42,
            )


            sft_trainer_signature = inspect.signature(SFTTrainer).parameters
            sft_trainer_args = train_params
            if "processing_class" in sft_trainer_signature:
                from trl import SFTConfig
                sft_config_kwargs = train_params.to_dict()
                sft_config_kwargs["hub_token"] = train_params.hub_token
                sft_config_kwargs.pop("push_to_hub_token", None)
                sft_config_kwargs["max_length"] = SEQUENCE_MAX_LENGTH_IN_TOKENS
                sft_trainer_args = SFTConfig(**sft_config_kwargs)

            sft_trainer_kwargs = {
                "model": model,
                "train_dataset": training_dataset,
                "eval_dataset": validation_dataset,
                "data_collator": data_collator,
                "args": sft_trainer_args,
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
            if preserve_epoch_checkpoints:
                trainer.add_callback(PreserveEpochCheckpointCallback(preserve_epoch_checkpoints))
                logging.info("Configured preserved epoch checkpoints: %s", preserve_epoch_checkpoints)


            ######################################################## TRAINING ########################################################


            logging.info("Start training")
            if trainer_resume_checkpoint is not None:
                resume_checkpoint_path = Path(trainer_resume_checkpoint)
                if not resume_checkpoint_path.exists():
                    raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_checkpoint_path}")
                logging.info("Resume trainer state from checkpoint: %s", resume_checkpoint_path)
                trainer.train(resume_from_checkpoint=str(resume_checkpoint_path))
            else:
                if adapter_init_checkpoint is not None:
                    logging.info(
                        "Training from adapter-initialized checkpoint %s for %s optimizer steps",
                        adapter_init_checkpoint,
                        effective_max_steps,
                    )
                trainer.train()

            if use_lora:
                finetune_model_path = build_lora_adapter_path(experiment.model_path)
            else:
                finetune_model_path = experiment.model_path + "fine_tuned_full"

            if deepspeed_config and use_lora and not use_unsloth:
                trainer.save_model(finetune_model_path)
            elif is_main_process:
                if use_unsloth:
                    from pipeline.unsloth_helpers import save_unsloth_model
                    save_unsloth_model(model, dp.tokenizer, finetune_model_path)
                elif use_lora:
                    model.save_pretrained(finetune_model_path)
                else:
                    model.save_pretrained(finetune_model_path)

            if is_distributed:
                torch.distributed.barrier()
                if not is_main_process:
                    return

            if skip_eval:
                if is_main_process and wandb.run is not None:
                    wandb.run.finish()
                logging.info("Skipping eval after training because skip_eval=True")
                return

            if not is_main_process:
                return


            # Clear GPU memory
            model = None
            del model
            gc.collect()
            torch.cuda.empty_cache()


            ######################################################## RELOAD MODEL ########################################################


            # For better predictions, we reload the model and adapter in 16 bit
            logging.info("Model reload")

            if use_unsloth:
                from pipeline.unsloth_helpers import load_unsloth_model_for_inference
                model, _unsloth_tokenizer = load_unsloth_model_for_inference(
                    MODEL_HF_NAME,
                    finetune_model_path,
                    max_seq_length=SEQUENCE_MAX_LENGTH_IN_TOKENS,
                )
            else:
                inference_load_kwargs = get_model_load_kwargs(
                    experiment.model_cache_path,
                    training=False,
                )
                if use_lora:
                    model = load_lora_model_for_inference(
                        MODEL_HF_NAME,
                        finetune_model_path,
                        inference_load_kwargs,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        finetune_model_path,
                        **inference_load_kwargs,
                    )


        if eval_model_path is not None and eval_backend == "hf":

            logging.info("Model load from path")

            if use_unsloth:
                from pipeline.unsloth_helpers import load_unsloth_model_for_inference
                model, _unsloth_tokenizer = load_unsloth_model_for_inference(
                    MODEL_HF_NAME,
                    eval_model_path,
                    max_seq_length=SEQUENCE_MAX_LENGTH_IN_TOKENS,
                )
            else:
                inference_load_kwargs = get_model_load_kwargs(
                    experiment.model_cache_path,
                    training=False,
                )
                if use_lora:
                    model = load_lora_model_for_inference(
                        MODEL_HF_NAME,
                        eval_model_path,
                        inference_load_kwargs,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        eval_model_path,
                        **inference_load_kwargs,
                    )
        elif eval_model_path is not None and eval_backend == "vllm":
            logging.info("Skipping local HF model load because eval backend is vllm")
            model = None


        # Setup data processing for inference
        dp.set_for_inference()


        ######################################################## EVALUATION SETUP ########################################################


        # Setup all pre and post processing functions
        logging.info("Setting up eval")

        def preprocessing_function(constants_row, true_events_input, true_future_events_input, target_dataframe, eval_manager):
            #: convert input to string
            str_input, str_output, meta_data = dp.convert_to_string_single_patient(constants_row, true_events_input, true_future_events_input,
                                                                                    target_dataframe, input_filtering_function=filtering_rows_rest_budget,
                                                                                    prompt=prompt, decimal_precision=DECIMAL_PRECISION,
                                                                                    constant_column_mapping=constant_column_mapping)
            return str_input, str_output, meta_data



        def encoding_function(input_text):

            #: apply pre-processing
            preprocessed_input_text = dp.preprocess_inputs([input_text])[0]

            return preprocessed_input_text



        def decoding_function(predictions):
            return dp.tokenizer.batch_decode(predictions, skip_special_tokens=True)



        def post_processing_function(prediction_string, patientid, patient_sample_index, meta_data):

            #: post process
            prediction_post = dp.post_process_string(prediction_string)

            if verbose:
                logging.info("Post processed predictions: " + prediction_post)

            #: convert all to dataframe
            all_days = meta_data["all_days_output"]
            predicted_df = dp.convert_to_df_single_patient(prediction_post, all_days, patientid, patient_sample_index, prediction_days_column_wise=meta_data["prediction_columns"])

            #: try convert to float here
            lab_vars = predicted_df.columns.str.contains("lab_")
            predicted_df.loc[:, lab_vars] = predicted_df.loc[:, lab_vars].apply(pd.to_numeric, errors='raise')

            # Return predicted DF
            return predicted_df


        experiment.model = model
        only_standardize = Only_Standardization(path_to_statistics_file)
        metric_manager = MetricManager(path_to_statistics_file)


        def evaluate_and_record(eval_set_events, eval_set_name, eval_meta_data, num_samples_to_generate, sample_merging_strategy, batch_size=BATCH_SIZE_VALIDATION):

            if eval_backend == "vllm":
                eval_targets, eval_prediction, return_meta_data_list = experiment.get_output_for_split_vllm_completions(
                    eval_set_events,
                    eval_manager,
                    preprocessing_function=preprocessing_function,
                    encoding_function=encoding_function,
                    post_processing_function=post_processing_function,
                    verbose=verbose,
                    max_new_tokens=max_new_tokens_to_generate,
                    num_samples_to_generate=num_samples_to_generate,
                    sample_merging_strategy=sample_merging_strategy,
                    max_output_length=4000,
                    return_meta_data=True,
                    prediction_url=prediction_url,
                    model_name=vllm_model_name or eval_model_path,
                    max_concurrent_requests=max_concurrent_requests,
                    temperature=vllm_temperature,
                    top_p=vllm_top_p,
                    tokenizer=dp.tokenizer,
                    total_max_length=vllm_total_max_length,
                    dynamic_max_tokens=vllm_dynamic_max_tokens,
                    minimum_max_tokens=vllm_minimum_max_tokens,
                    fail_on_request_error=vllm_fail_on_request_error,
                )
            else:
                eval_targets, eval_prediction, return_meta_data_list = experiment.get_output_for_split_hf_default(eval_set_events,
                                                                                            eval_manager,
                                                                                            preprocessing_function=preprocessing_function,
                                                                                            tokenizer=dp.tokenizer,
                                                                                            encoding_function=encoding_function,
                                                                                            decoding_function=decoding_function,
                                                                                            post_processing_function=post_processing_function,
                                                                                            batch_size=batch_size,
                                                                                            verbose=verbose,
                                                                                            gen_top_p=0.9,
                                                                                            gen_do_sample=True,
                                                                                            max_new_tokens=max_new_tokens_to_generate,
                                                                                            num_samples_to_generate=num_samples_to_generate,
                                                                                            sample_merging_strategy=sample_merging_strategy,
                                                                                            pad_token_id=dp.tokenizer.eos_token_id,
                                                                                            max_output_length=4000,                     # Setting this here due to mistral long context bug, need to check if causes errors
                                                                                            return_meta_data=True,
                                                                                            note_down_probabilities=True)

            # Do filtering without standardizing
            eval_targets_filtered, eval_prediction_filtered = only_standardize.normalize_and_filter(eval_targets, eval_prediction)

            #: set grouping by therapy
            eval_targets_filtered_with_meta_data = experiment.join_meta_data_to_targets(eval_targets_filtered, eval_meta_data, generated_meta_data=return_meta_data_list)

            # Calculate performance metrics
            eval_performance = metric_manager.calculate_metrics(eval_targets_filtered, eval_prediction_filtered, group_by=None)

            # Save tables locally and record in wandb
            experiment.save_df_targets_predictions_locally_and_statistics_to_wandb(eval_set_name, eval_targets_filtered, eval_prediction_filtered, meta_data_df=eval_targets_filtered_with_meta_data)

            # Save performance to wandb
            experiment.save_to_wandb_final_performances(eval_performance, eval_set_name)

            return eval_targets_filtered, eval_prediction_filtered, eval_targets_filtered_with_meta_data



        ######################################################## TEST EVAL ########################################################

        logging.info("Test set Eval")
        test_targets, test_prediction, test_meta_data = evaluate_and_record(test_events, test_set_output_name, test_meta, num_samples_to_generate=num_samples_to_generate, sample_merging_strategy=sample_merging_strategy)



        ############################################################ Finish run ############################################################
        wandb.run.finish()
