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


def check_vllm_endpoint(prediction_url, timeout=10, expected_model_name=None):
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
    except urllib.error.URLError as error:
        raise VLLMEndpointError(
            f"vLLM endpoint is not reachable at {models_url}: {error}. "
            "Start the vLLM OpenAI server on this host/port or pass --prediction-url."
        ) from error
    except Exception as error:
        raise VLLMEndpointError(
            f"vLLM endpoint returned an invalid /models response at {models_url}: {error}"
        ) from error

    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        raise VLLMEndpointError(
            f"vLLM endpoint returned an invalid /models response at {models_url}: expected a JSON object with a data list."
        )

    model_ids = [model.get("id") for model in body["data"] if isinstance(model, dict) and model.get("id")]
    if not model_ids:
        raise VLLMEndpointError(
            f"vLLM endpoint is reachable at {models_url}, but it did not report any model IDs."
        )
    if expected_model_name is not None and expected_model_name not in model_ids:
        raise VLLMEndpointError(
            f"vLLM endpoint at {models_url} does not serve requested model {expected_model_name!r}; "
            f"available models: {model_ids}"
        )
    return model_ids


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
