import argparse
import os
import sys
from pathlib import Path

# To overcome issues with CUDA OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import __init__
from pipeline.local_paths import ensure_runtime_cache_env
from dt_gpt_fft_2024_04_11_biomistral_td_bd_sr import DTGPT_mimic_biomistral_fft_ti_bd_sr

ensure_runtime_cache_env()

if not hasattr(argparse, "BooleanOptionalAction"):
    class _BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, **kwargs):
            option_strings = list(option_strings)
            extended_option_strings = []
            for option_string in option_strings:
                extended_option_strings.append(option_string)
                if option_string.startswith("--"):
                    extended_option_strings.append("--no-" + option_string[2:])
            super().__init__(
                option_strings=extended_option_strings,
                dest=dest,
                nargs=0,
                default=default,
                **kwargs,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not option_string.startswith("--no-"))

    argparse.BooleanOptionalAction = _BooleanOptionalAction


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def parse_epoch_list(value):
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return []
    epochs = []
    for token in stripped.split(","):
        parsed = int(token.strip())
        if parsed < 1:
            raise argparse.ArgumentTypeError("epoch values must be >= 1")
        epochs.append(parsed)
    return epochs


def build_parser():
    parser = argparse.ArgumentParser(description="Train DT-GPT on MIMIC-IV with BioMistral.")
    parser.add_argument("--debug", action="store_true", help="Disable WandB logging.")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=10)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=float, default=5)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument(
        "--preserve-epoch-checkpoints",
        type=parse_epoch_list,
        default=None,
        help="Comma-separated epoch numbers whose checkpoints should be copied outside HF rotation.",
    )
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
    parser.add_argument("--use-dora", action="store_true")
    parser.add_argument("--use-unsloth", action="store_true",
                        help="Use Unsloth for optimized 4-bit QLoRA+DoRA training. "
                             "Implies --use-lora and --use-dora.")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--sft-dataset-num-proc",
        type=int,
        default=1,
        help="Number of processes TRL should use for SFT dataset preprocessing.",
    )
    parser.add_argument(
        "--df-conversion-n-jobs",
        type=int,
        default=None,
        help="Number of joblib workers for DF-to-string conversion. Defaults to DTGPT_DF_CONVERSION_N_JOBS or 1.",
    )
    parser.add_argument("--deepspeed-config", type=str, default=None)
    parser.add_argument("--train-max-patients", type=int, default=None)
    parser.add_argument("--validation-max-patients", type=int, default=None)
    parser.add_argument("--test-max-patients", type=int, default=None)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--validation-max-samples", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument("--prediction-url", type=str, default="http://127.0.0.1:18101/v1/")
    parser.add_argument("--vllm-model-name", type=str, default=None)
    parser.add_argument("--max-concurrent-requests", type=positive_int, default=16)
    parser.add_argument("--vllm-fail-on-request-error", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main():
    args = build_parser().parse_args()

    experiment = DTGPT_mimic_biomistral_fft_ti_bd_sr()

    experiment.run(
        debug=args.debug,
        verbose=True,
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
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        preserve_epoch_checkpoints=args.preserve_epoch_checkpoints,
        eval_interval=args.eval_interval,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler="cosine",
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        sft_dataset_num_proc=args.sft_dataset_num_proc,
        df_conversion_n_jobs=args.df_conversion_n_jobs,
        train_max_patients=args.train_max_patients,
        validation_max_patients=args.validation_max_patients,
        test_max_patients=args.test_max_patients,
        train_max_samples=args.train_max_samples,
        validation_max_samples=args.validation_max_samples,
        skip_eval=args.skip_eval,
        nr_days_forecasting=91,
        seq_max_len_in_tokens=args.seq_max_len,
        decimal_precision=args.decimal_precision,
        num_samples_to_generate=args.num_samples_to_generate,
        sample_merging_strategy=args.sample_merging_strategy,
        max_new_tokens_to_generate=args.max_new_tokens_to_generate,
        use_lora=args.use_lora or args.use_dora or args.use_unsloth,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_dora=args.use_dora or args.use_unsloth,
        use_unsloth=args.use_unsloth,
        deepspeed_config=args.deepspeed_config,
        prediction_url=args.prediction_url,
        vllm_model_name=args.vllm_model_name,
        max_concurrent_requests=args.max_concurrent_requests,
        vllm_fail_on_request_error=args.vllm_fail_on_request_error,
    )


if __name__ == "__main__":
    main()
