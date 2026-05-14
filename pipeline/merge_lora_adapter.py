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
    return parser


def main():
    args = build_parser().parse_args()

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    load_kwargs = get_model_load_kwargs(args.cache_dir, training=False)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=False)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(output_path)

    print("Merged model saved to: " + str(output_path))


if __name__ == "__main__":
    main()
