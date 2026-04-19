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
    parser = argparse.ArgumentParser(description="Evaluate a DT-GPT MIMIC-IV checkpoint.")
    parser.add_argument("--eval-model-path", required=True, help="Checkpoint or model directory to evaluate.")
    parser.add_argument("--debug", action="store_true", help="Disable WandB logging.")
    parser.add_argument("--validation-batch-size", type=int, default=5)
    parser.add_argument("--seq-max-len", type=int, default=3400)
    parser.add_argument("--num-samples-to-generate", type=int, default=30)
    parser.add_argument("--max-new-tokens-to-generate", type=int, default=900)
    return parser


def main():
    args = build_parser().parse_args()

    experiment = DTGPT_mimic_biomistral_fft_ti_bd_sr()

    experiment.run(
        debug=args.debug,
        verbose=True,
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


if __name__ == "__main__":
    main()
