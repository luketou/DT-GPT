import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.local_paths import get_model_load_kwargs


def build_parser():
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA/DoRA adapter into a full HF model.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--cache-dir", default="3_cache")
    parser.add_argument(
        "--device-map",
        default="auto",
        choices=("auto", "none"),
        help="Use 'none' to avoid PEFT offload/meta loading issues while merging.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    device_map = None if args.device_map == "none" else args.device_map
    load_kwargs = get_model_load_kwargs(args.cache_dir, device_map=device_map, training=False)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    if device_map is None:
        import torch

        if torch.cuda.is_available():
            base_model = base_model.to("cuda")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=False)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(output_path)

    print("Merged model saved to: " + str(output_path))


if __name__ == "__main__":
    main()
