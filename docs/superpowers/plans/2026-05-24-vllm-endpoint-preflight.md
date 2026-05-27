# vLLM Endpoint Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fail MIMIC vLLM evaluation early with a clear endpoint diagnostic instead of launching thousands of doomed requests when `http://127.0.0.1:18101/v1/` is not serving.

**Architecture:** Extract minimal vLLM HTTP helpers into `pipeline/vllm_client.py`, add a preflight `/models` check before request fan-out in `Experiment.get_output_for_split_vllm_completions`, and expose existing vLLM settings through the MIMIC runner CLI so batch scripts can point to the correct server/port. Keep behavior unchanged after a healthy preflight; do not add dependencies.

**Tech Stack:** Python stdlib (`urllib`, `json`, `socket`, `time`), existing `pytest`/`compileall` validation, existing MIMIC experiment scripts.

---

## Root-cause audit summary

Evidence reviewed:
- `logs/mimic_dora_resume1395_to4185_37667.out`: vLLM eval starts `2026-05-24 12:04:48`, first request fails at `12:04:49`, and all `2787` request lines fail with `Connection refused`.
- `logs/mimic_dora_r32_minival_37738.out`: vLLM eval starts `2026-05-24 15:54:22`, first request fails at `15:54:22`, and all `2787` request lines fail with `Connection refused`.
- Both `.err` files end at `pipeline/Experiment.py:891` inside `urllib.request.urlopen(...)`, raised as `urllib.error.URLError: <urlopen error [Errno 111] Connection refused>`.
- `dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py:101` hardcodes `prediction_url="http://127.0.0.1:18101/v1/"` unless the caller passes a different value.
- `2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py` currently does not expose `prediction_url`, `max_concurrent_requests`, or vLLM fail/continue behavior as CLI arguments.

Conclusion:
- Immediate root cause: no HTTP server was accepting connections at `127.0.0.1:18101` when evaluation began.
- Code-level contributing cause: client code has no vLLM health preflight, so it schedules all 2787 requests and only reports the aggregate failure after flooding logs.
- Operational contributing cause: the runnable script does not make the endpoint configurable from CLI, increasing the chance of mismatched Slurm/vLLM port assignments.

---

## File structure

- Create: `pipeline/vllm_client.py`
  - Responsibility: small, testable stdlib helper functions for vLLM URL normalization, `/models` preflight, and `/completions` POST.
- Modify: `pipeline/Experiment.py`
  - Responsibility: call preflight once before `generate_all()` fan-out and use helper for completion POST.
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py`
  - Responsibility: expose vLLM endpoint/concurrency/fail behavior as CLI args and pass them into `experiment.run(...)`.
- Create: `tests/test_vllm_client.py`
  - Responsibility: unit tests for URL construction, preflight failure message, preflight success parsing, and POST payload behavior without running a model.

---

### Task 1: Add failing tests for vLLM URL and preflight behavior

**Files:**
- Create: `tests/test_vllm_client.py`

- [ ] **Step 1: Create the test file**

```python
import json
import urllib.error

import pytest

from pipeline.vllm_client import (
    VLLMEndpointError,
    build_vllm_url,
    check_vllm_endpoint,
)


def test_build_vllm_url_appends_path_once():
    assert build_vllm_url("http://127.0.0.1:18101/v1/", "models") == "http://127.0.0.1:18101/v1/models"
    assert build_vllm_url("http://127.0.0.1:18101/v1", "/completions") == "http://127.0.0.1:18101/v1/completions"


def test_check_vllm_endpoint_wraps_connection_failure(monkeypatch):
    def fail_urlopen(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)

    with pytest.raises(VLLMEndpointError) as excinfo:
        check_vllm_endpoint("http://127.0.0.1:18101/v1/", timeout=1)

    message = str(excinfo.value)
    assert "vLLM endpoint is not reachable" in message
    assert "http://127.0.0.1:18101/v1/models" in message
    assert "Connection refused" in message


def test_check_vllm_endpoint_accepts_models_response(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"data": [{"id": "local-model"}]}).encode("utf-8")

    def ok_urlopen(request, timeout):
        assert request.full_url == "http://127.0.0.1:18101/v1/models"
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", ok_urlopen)

    assert check_vllm_endpoint("http://127.0.0.1:18101/v1/", timeout=1) == ["local-model"]
```

- [ ] **Step 2: Run test to verify it fails before implementation**

Run:

```bash
pytest tests/test_vllm_client.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'pipeline.vllm_client'
```

---

### Task 2: Implement minimal vLLM client helpers

**Files:**
- Create: `pipeline/vllm_client.py`

- [ ] **Step 1: Add helper implementation**

```python
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
```

- [ ] **Step 2: Run tests**

Run:

```bash
pytest tests/test_vllm_client.py -v
```

Expected:

```text
3 passed
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/vllm_client.py tests/test_vllm_client.py
git commit -m "Fail early when vLLM endpoint is unreachable

Constraint: MIMIC evaluation depends on an external OpenAI-compatible vLLM server.
Rejected: Only lowering request concurrency | It would still fail when no process listens on the configured port.
Confidence: high
Scope-risk: narrow
Tested: pytest tests/test_vllm_client.py -v"
```

---

### Task 3: Add preflight to `Experiment.get_output_for_split_vllm_completions`

**Files:**
- Modify: `pipeline/Experiment.py`

- [ ] **Step 1: Replace local URL construction/post code with helper usage**

At the top of `get_output_for_split_vllm_completions`, replace local imports:

```python
        import urllib.error
        import urllib.request
```

with:

```python
        from pipeline.vllm_client import check_vllm_endpoint, post_vllm_completion
```

Inside `generate_all()`, delete:

```python
            completions_url = prediction_url.rstrip("/") + "/completions"
```

and replace `post_completion` with:

```python
            def post_completion(payload):
                return post_vllm_completion(prediction_url, payload, timeout=600)
```

- [ ] **Step 2: Add preflight before `asyncio.run(generate_all())`**

Immediately before:

```python
        if requests:
            for patientid, patient_sample_index, prediction_str in asyncio.run(generate_all()):
```

insert:

```python
        if requests:
            available_models = check_vllm_endpoint(prediction_url)
            logging.info(
                "vLLM endpoint preflight passed for "
                + str(prediction_url)
                + "; available models: "
                + str(available_models)
            )
            for patientid, patient_sample_index, prediction_str in asyncio.run(generate_all()):
```

Remove the old duplicate `if requests:` line so there is only one block.

- [ ] **Step 3: Run targeted tests**

Run:

```bash
pytest tests/test_vllm_client.py -v
python -m compileall pipeline/Experiment.py pipeline/vllm_client.py
```

Expected:

```text
3 passed
...
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/Experiment.py pipeline/vllm_client.py tests/test_vllm_client.py
git commit -m "Stop vLLM evaluation before unreachable fan-out

Constraint: A missing vLLM server currently creates thousands of request failures before surfacing the root error.
Rejected: Keep aggregate failure only | It obscures the real endpoint configuration problem and bloats logs.
Confidence: high
Scope-risk: narrow
Tested: pytest tests/test_vllm_client.py -v; python -m compileall pipeline/Experiment.py pipeline/vllm_client.py"
```

---

### Task 4: Expose vLLM endpoint controls in the MIMIC runner CLI

**Files:**
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py`

- [ ] **Step 1: Add parser arguments after `--skip-eval`**

```python
    parser.add_argument("--prediction-url", type=str, default="http://127.0.0.1:18101/v1/")
    parser.add_argument("--vllm-model-name", type=str, default=None)
    parser.add_argument("--max-concurrent-requests", type=int, default=16)
    parser.add_argument("--vllm-fail-on-request-error", action=argparse.BooleanOptionalAction, default=True)
```

- [ ] **Step 2: Pass the arguments into `experiment.run(...)`**

Add these keyword arguments in the existing `experiment.run(...)` call:

```python
        prediction_url=args.prediction_url,
        vllm_model_name=args.vllm_model_name,
        max_concurrent_requests=args.max_concurrent_requests,
        vllm_fail_on_request_error=args.vllm_fail_on_request_error,
```

- [ ] **Step 3: Validate parser syntax**

Run:

```bash
python -m compileall 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py
```

Expected:

```text
Compiling '.../2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py'...
```

- [ ] **Step 4: Commit**

```bash
git add 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py
git commit -m "Make MIMIC vLLM endpoint configurable

Constraint: Slurm jobs may launch vLLM on different host/port combinations.
Rejected: Editing hardcoded defaults per run | CLI parameters are safer and reproducible in logs.
Confidence: high
Scope-risk: narrow
Tested: python -m compileall 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py"
```

---

### Task 5: Add a smoke command for endpoint diagnosis before long jobs

**Files:**
- Create: `scripts/check_vllm_endpoint.py`

- [ ] **Step 1: Add smoke-check script**

```python
import argparse

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
```

- [ ] **Step 2: Run syntax validation**

Run:

```bash
python -m compileall scripts/check_vllm_endpoint.py
```

Expected:

```text
Compiling 'scripts/check_vllm_endpoint.py'...
```

- [ ] **Step 3: Document operational usage in the plan or Slurm notes**

Before launching a long job, run:

```bash
python scripts/check_vllm_endpoint.py --prediction-url http://127.0.0.1:18101/v1/
```

Expected if server is down:

```text
pipeline.vllm_client.VLLMEndpointError: vLLM endpoint is not reachable at http://127.0.0.1:18101/v1/models: <urlopen error [Errno 111] Connection refused>. Start the vLLM OpenAI server on this host/port or pass --prediction-url.
```

Expected if server is up:

```text
vLLM endpoint OK: http://127.0.0.1:18101/v1/
Models: ['...']
```

- [ ] **Step 4: Commit**

```bash
git add scripts/check_vllm_endpoint.py
git commit -m "Add vLLM endpoint smoke check

Constraint: Failed jobs spend hours training before discovering the eval endpoint is down.
Rejected: Rely only on runtime preflight | A standalone smoke command helps Slurm scripts fail before expensive work.
Confidence: medium
Scope-risk: narrow
Tested: python -m compileall scripts/check_vllm_endpoint.py"
```

---

## Final verification

- [ ] Run unit tests:

```bash
pytest tests/test_vllm_client.py -v
```

- [ ] Run syntax checks:

```bash
python -m compileall pipeline 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py scripts/check_vllm_endpoint.py
```

- [ ] Reproduce the current failure safely with no vLLM server on `18101`:

```bash
python scripts/check_vllm_endpoint.py --prediction-url http://127.0.0.1:18101/v1/
```

Expected: one clear `VLLMEndpointError`, not thousands of request failures.

- [ ] When vLLM server is available, run:

```bash
python scripts/check_vllm_endpoint.py --prediction-url http://127.0.0.1:18101/v1/
```

Expected: `vLLM endpoint OK` with model IDs.

## Self-review

- Spec coverage: covers root-cause audit, endpoint preflight, clearer failure, CLI configurability, and smoke validation.
- Placeholder scan: no deferred-work markers or unspecified test steps remain.
- Type consistency: helper names are consistent across tests, implementation, and call sites.
