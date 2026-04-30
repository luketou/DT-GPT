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
    parser.add_argument("--deepspeed-config", type=str, default=None)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
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
        eval_interval=args.eval_interval,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler="cosine",
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        sft_dataset_num_proc=args.sft_dataset_num_proc,
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
    )


if __name__ == "__main__":
    main()
