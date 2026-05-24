import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.vllm_client import VLLMEndpointError, check_vllm_endpoint


def main():
    parser = argparse.ArgumentParser(description="Check a vLLM OpenAI-compatible endpoint before launching evaluation.")
    parser.add_argument("--prediction-url", default="http://127.0.0.1:18101/v1/")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--expected-model", default=None)
    args = parser.parse_args()

    try:
        models = check_vllm_endpoint(
            args.prediction_url,
            timeout=args.timeout,
            expected_model_name=args.expected_model,
        )
    except VLLMEndpointError as error:
        print(str(error), file=sys.stderr)
        return 1

    print("vLLM endpoint OK:", args.prediction_url)
    print("Models:", models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
