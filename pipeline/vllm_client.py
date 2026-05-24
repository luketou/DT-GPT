import json
import os
import urllib.error
import urllib.request


class VLLMEndpointError(RuntimeError):
    """Raised when the configured vLLM OpenAI-compatible endpoint is unusable."""


def build_vllm_url(base_url, path):
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _authorization_header():
    return "Bearer " + os.environ.get("DTGPT_VLLM_API_KEY", "token-abc123")


def check_vllm_endpoint(prediction_url, timeout=10):
    models_url = build_vllm_url(prediction_url, "models")
    request = urllib.request.Request(
        models_url,
        headers={"Authorization": _authorization_header()},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise VLLMEndpointError(
            f"vLLM endpoint is not usable at {models_url}: HTTP {error.code}: {error_body}"
        ) from error
    except Exception as error:
        raise VLLMEndpointError(
            f"vLLM endpoint is not reachable at {models_url}: {error}. "
            "Start the vLLM OpenAI server on this host/port or pass --prediction-url."
        ) from error

    return [model.get("id") for model in body.get("data", []) if isinstance(model, dict) and model.get("id")]


def post_vllm_completion(prediction_url, payload, timeout=600):
    completions_url = build_vllm_url(prediction_url, "completions")
    request = urllib.request.Request(
        completions_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": _authorization_header(),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {error.code} for {completions_url}: {error_body}") from error
