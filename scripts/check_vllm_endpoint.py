import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.vllm_client import check_vllm_endpoint


def main():
    parser = argparse.ArgumentParser(description="Check a vLLM OpenAI-compatible endpoint before launching evaluation.")
    parser.add_argument("--prediction-url", default="http://127.0.0.1:18101/v1/")
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()

    models = check_vllm_endpoint(args.prediction_url, timeout=args.timeout)
    print("vLLM endpoint OK:", args.prediction_url)
    print("Models:", models)


if __name__ == "__main__":
    main()
