# vLLM Dynamic Generation Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent MIMIC vLLM evaluation requests from failing when `prompt_tokens + max_tokens` exceeds the model context window.

**Architecture:** Keep prompt construction unchanged and fix the vLLM request layer so each request gets a safe output-token budget computed from the actual encoded prompt length. The reusable budget helper lives in `pipeline/Experiment.py`; the MIMIC eval wrapper exposes CLI/env knobs and passes the tokenizer into the vLLM evaluation path.

**Tech Stack:** Python, stdlib `unittest`, Hugging Face tokenizer interface, vLLM OpenAI-compatible `/v1/completions`, existing DT-GPT `Experiment` evaluation pipeline.

---

## Log diagnosis

The tail of `/home/r15543056/trajectory_forecast/DT-GPT/logs/mimic_dora_vllm_server_36949_7.log` ends with repeated vLLM validation failures:

```text
vllm.exceptions.VLLMValidationError: This model's maximum context length is 4096 tokens. However, you requested 900 output tokens and your prompt contains at least 3197 input tokens, for a total of at least 4097 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=3197)
INFO:     127.0.0.1:47812 - "POST /v1/completions HTTP/1.1" 400 Bad Request
```

Root cause in current code:

- `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py:27` defaults `--max-new-tokens-to-generate` to `900`.
- `pipeline/Experiment.py:850-856` sends that same fixed `max_new_tokens` as vLLM `max_tokens` for every request.
- For the failing request, vLLM sees `3197 + 900 = 4097`, which is one token over the 4096 context window.

Target behavior:

- For a prompt with 3197 tokens, requested output 900 tokens, and safe total budget 4092, send `max_tokens=895` because `3197 + 895 = 4092 < 4096`.
- For short prompts, keep the requested limit, e.g. prompt 1000 + requested 900 => `max_tokens=900`.
- Log the per-run min/max dynamic budgets so future failures can be diagnosed from client logs.

## File structure

- Modify `pipeline/Experiment.py`
  - Add `compute_vllm_max_tokens(...)` helper near the existing top-level helpers.
  - Extend `Experiment.get_output_for_split_vllm_completions(...)` with tokenizer and vLLM budget options.
  - Store request-specific `max_tokens` and send that in the HTTP payload.
  - Improve request failure logging to include request index and computed `max_tokens`.
- Modify `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
  - Add vLLM dynamic-budget parameters to `DTGPT_mimic_biomistral_fft_ti_bd_sr.run(...)`.
  - Pass `dp.tokenizer` and these options into `get_output_for_split_vllm_completions(...)`.
- Modify `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py`
  - Add CLI/env options for dynamic token budgeting.
  - Default dynamic budgeting on for this MIMIC vLLM eval entrypoint.
- Create `tests/test_vllm_generation_budget.py`
  - Unit-test the helper without loading a real tokenizer or model.

### Task 1: Add failing tests for vLLM output-token budgeting

**Files:**
- Create: `tests/test_vllm_generation_budget.py`
- Test: `tests/test_vllm_generation_budget.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_vllm_generation_budget.py` with this exact content:

```python
import unittest

from pipeline.Experiment import compute_vllm_max_tokens


class FakeTokenizerResult:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class FakeTokenizer:
    def __init__(self, token_count):
        self.token_count = token_count

    def __call__(self, text, add_special_tokens=False):
        return FakeTokenizerResult(list(range(self.token_count)))


class VllmGenerationBudgetTests(unittest.TestCase):
    def test_dynamic_budget_caps_requested_new_tokens(self):
        tokenizer = FakeTokenizer(token_count=3200)

        max_tokens = compute_vllm_max_tokens(
            prompt="ignored by fake tokenizer",
            tokenizer=tokenizer,
            requested_max_new_tokens=900,
            total_max_length=4092,
            dynamic_max_tokens=True,
            minimum_max_tokens=1,
        )

        self.assertEqual(max_tokens, 892)

    def test_dynamic_budget_uses_requested_limit_when_prompt_is_short(self):
        tokenizer = FakeTokenizer(token_count=1000)

        max_tokens = compute_vllm_max_tokens(
            prompt="ignored by fake tokenizer",
            tokenizer=tokenizer,
            requested_max_new_tokens=900,
            total_max_length=4092,
            dynamic_max_tokens=True,
            minimum_max_tokens=1,
        )

        self.assertEqual(max_tokens, 900)

    def test_dynamic_budget_never_returns_less_than_minimum(self):
        tokenizer = FakeTokenizer(token_count=4092)

        max_tokens = compute_vllm_max_tokens(
            prompt="ignored by fake tokenizer",
            tokenizer=tokenizer,
            requested_max_new_tokens=900,
            total_max_length=4092,
            dynamic_max_tokens=True,
            minimum_max_tokens=1,
        )

        self.assertEqual(max_tokens, 1)

    def test_static_budget_keeps_requested_new_tokens(self):
        tokenizer = FakeTokenizer(token_count=4092)

        max_tokens = compute_vllm_max_tokens(
            prompt="ignored by fake tokenizer",
            tokenizer=tokenizer,
            requested_max_new_tokens=768,
            total_max_length=4092,
            dynamic_max_tokens=False,
            minimum_max_tokens=1,
        )

        self.assertEqual(max_tokens, 768)

    def test_dynamic_budget_requires_tokenizer(self):
        with self.assertRaises(ValueError):
            compute_vllm_max_tokens(
                prompt="abc",
                tokenizer=None,
                requested_max_new_tokens=900,
                total_max_length=4092,
                dynamic_max_tokens=True,
                minimum_max_tokens=1,
            )

    def test_dynamic_budget_matches_failed_36949_context_case(self):
        tokenizer = FakeTokenizer(token_count=3197)

        max_tokens = compute_vllm_max_tokens(
            prompt="ignored by fake tokenizer",
            tokenizer=tokenizer,
            requested_max_new_tokens=900,
            total_max_length=4092,
            dynamic_max_tokens=True,
            minimum_max_tokens=1,
        )

        self.assertEqual(max_tokens, 895)
        self.assertLessEqual(3197 + max_tokens, 4092)
        self.assertLess(3197 + max_tokens, 4096)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new test and verify it fails for the expected reason**

Run:

```bash
python -m unittest tests.test_vllm_generation_budget -v
```

Expected: FAIL or ERROR with an import error like:

```text
ImportError: cannot import name 'compute_vllm_max_tokens' from 'pipeline.Experiment'
```

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_vllm_generation_budget.py
git commit -m "Add coverage for vLLM token budget overflow

Constraint: vLLM rejects prompt plus output budgets that exceed the model context window.
Confidence: high
Scope-risk: narrow
Tested: python -m unittest tests.test_vllm_generation_budget -v fails before implementation because compute_vllm_max_tokens is missing
Not-tested: full MIMIC vLLM evaluation"
```

### Task 2: Implement the reusable vLLM token-budget helper

**Files:**
- Modify: `pipeline/Experiment.py:29-61`
- Test: `tests/test_vllm_generation_budget.py`

- [ ] **Step 1: Add `compute_vllm_max_tokens` after `align_prediction_to_target_shape`**

In `pipeline/Experiment.py`, insert this function after the existing `align_prediction_to_target_shape(...)` function and before `class Experiment:`:

```python
def compute_vllm_max_tokens(
    prompt,
    tokenizer,
    requested_max_new_tokens,
    total_max_length,
    dynamic_max_tokens,
    minimum_max_tokens=1,
):
    """Return the vLLM output-token budget for one prompt.

    HF generation in the original DT-GPT eval used a total `max_length`
    budget. vLLM's OpenAI-compatible completions endpoint instead expects
    output-only `max_tokens`, so dynamic mode converts total budget to
    output budget per prompt.
    """
    if not dynamic_max_tokens:
        return int(requested_max_new_tokens)

    if tokenizer is None:
        raise ValueError("tokenizer is required when dynamic_max_tokens=True")

    tokenized_prompt = tokenizer(prompt, add_special_tokens=False)
    prompt_token_count = len(tokenized_prompt.input_ids)
    remaining_tokens = int(total_max_length) - prompt_token_count
    bounded_remaining_tokens = max(int(minimum_max_tokens), remaining_tokens)
    return min(int(requested_max_new_tokens), bounded_remaining_tokens)
```

- [ ] **Step 2: Run the focused unit tests**

Run:

```bash
python -m unittest tests.test_vllm_generation_budget -v
```

Expected:

```text
Ran 6 tests
OK
```

- [ ] **Step 3: Commit the helper**

```bash
git add pipeline/Experiment.py tests/test_vllm_generation_budget.py
git commit -m "Bound vLLM output tokens by prompt length

Constraint: BioMistral vLLM serves a 4096-token context and rejects request-level overflow.
Rejected: Lower the global default max-new-tokens only | it wastes budget for short prompts and still fails if prompts vary.
Confidence: high
Scope-risk: narrow
Directive: Keep vLLM max_tokens as output-only tokens; do not treat it as total max_length.
Tested: python -m unittest tests.test_vllm_generation_budget -v
Not-tested: live vLLM server request"
```

### Task 3: Wire per-request budgets into `get_output_for_split_vllm_completions`

**Files:**
- Modify: `pipeline/Experiment.py:734-873`
- Test: `tests/test_vllm_generation_budget.py`

- [ ] **Step 1: Extend the method signature**

Change the `get_output_for_split_vllm_completions(...)` signature in `pipeline/Experiment.py` from:

```python
                                              temperature=1.0,
                                              top_p=0.9,
                                              num_samples_to_generate=1,
```

to:

```python
                                              temperature=1.0,
                                              top_p=0.9,
                                              tokenizer=None,
                                              total_max_length=4092,
                                              dynamic_max_tokens=False,
                                              minimum_max_tokens=1,
                                              fail_on_request_error=True,
                                              num_samples_to_generate=1,
```

- [ ] **Step 2: Store request-specific max tokens when building `requests`**

Replace the current `for sample_idx in range(num_samples_to_generate): requests.append(...)` block around `pipeline/Experiment.py:790-796` with:

```python
                for sample_idx in range(num_samples_to_generate):
                    request_max_tokens = compute_vllm_max_tokens(
                        prompt=str_input,
                        tokenizer=tokenizer,
                        requested_max_new_tokens=max_new_tokens,
                        total_max_length=total_max_length,
                        dynamic_max_tokens=dynamic_max_tokens,
                        minimum_max_tokens=minimum_max_tokens,
                    )
                    requests.append({
                        "patientid": patientid,
                        "patient_sample_index": patient_sample_index,
                        "prompt": str_input,
                        "seed": 8719 + (idx * max(num_samples_to_generate, 1)) + sample_idx,
                        "max_tokens": request_max_tokens,
                    })
```

- [ ] **Step 3: Log budget summary before `generate_all()`**

Insert this block after the request-building loop and before `async def generate_all():`:

```python
        if requests:
            request_token_budgets = [request["max_tokens"] for request in requests]
            logging.info(
                "vLLM max_tokens budget summary: min="
                + str(min(request_token_budgets))
                + ", max="
                + str(max(request_token_budgets))
                + ", dynamic="
                + str(dynamic_max_tokens)
                + ", total_max_length="
                + str(total_max_length)
            )
```

- [ ] **Step 4: Send per-request `max_tokens` in the HTTP payload**

Replace this payload field in `pipeline/Experiment.py:853`:

```python
                                "max_tokens": max_new_tokens,
```

with:

```python
                                "max_tokens": request["max_tokens"],
```

- [ ] **Step 5: Improve failed-request logging and optional continuation**

Replace the current exception handler and failed-request block:

```python
                    except Exception:
                        traceback.print_exc()
                        failed_requests.append(request_idx + 1)
                        logging.info("vLLM request failed.")
                        completion_text = ""
```

with:

```python
                    except Exception as exc:
                        traceback.print_exc()
                        failed_requests.append(request_idx + 1)
                        logging.info(
                            "vLLM request failed for index "
                            + str(request_idx + 1)
                            + " with max_tokens="
                            + str(request["max_tokens"])
                            + ": "
                            + str(exc)
                        )
                        completion_text = ""
```

Then replace:

```python
            if failed_requests:
                raise RuntimeError(
                    "vLLM generation failed for "
                    + str(len(failed_requests))
                    + " request(s); first failed request indices: "
                    + str(failed_requests[:20])
                )
```

with:

```python
            if failed_requests:
                message = (
                    "vLLM generation failed for "
                    + str(len(failed_requests))
                    + " request(s); first failed request indices: "
                    + str(failed_requests[:20])
                )
                if fail_on_request_error:
                    raise RuntimeError(message)
                logging.warning(message + "; continuing with empty failed completions")
```

- [ ] **Step 6: Run focused tests and syntax check**

Run:

```bash
python -m unittest tests.test_vllm_generation_budget -v
python -m compileall pipeline/Experiment.py
```

Expected:

```text
Ran 6 tests
OK
```

and compileall returns success without `SyntaxError`.

- [ ] **Step 7: Commit request wiring**

```bash
git add pipeline/Experiment.py tests/test_vllm_generation_budget.py
git commit -m "Use per-request vLLM generation budgets

Constraint: Request prompt lengths vary across MIMIC samples and seeds.
Rejected: Retry after HTTP 400 | it delays failure and still loses the original budget context.
Confidence: high
Scope-risk: moderate
Directive: Log request budget summaries before issuing concurrent vLLM calls.
Tested: python -m unittest tests.test_vllm_generation_budget -v; python -m compileall pipeline/Experiment.py
Not-tested: live vLLM server request"
```

### Task 4: Expose dynamic vLLM budget settings in the MIMIC experiment and wrapper

**Files:**
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py:70-78`
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py:589-607`
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py:20-78`
- Test: `tests/test_vllm_generation_budget.py`

- [ ] **Step 1: Add parameters to the experiment `run(...)` signature**

In `dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`, change the end of the `run(...)` signature from:

```python
                max_concurrent_requests=16,
                vllm_temperature=1.0,
                vllm_top_p=0.9):
```

to:

```python
                max_concurrent_requests=16,
                vllm_temperature=1.0,
                vllm_top_p=0.9,
                vllm_total_max_length=4092,
                vllm_dynamic_max_tokens=True,
                vllm_minimum_max_tokens=1,
                vllm_fail_on_request_error=True):
```

- [ ] **Step 2: Pass tokenizer and dynamic-budget settings into vLLM evaluation**

In the `experiment.get_output_for_split_vllm_completions(...)` call, after:

```python
                    temperature=vllm_temperature,
                    top_p=vllm_top_p,
```

add:

```python
                    tokenizer=dp.tokenizer,
                    total_max_length=vllm_total_max_length,
                    dynamic_max_tokens=vllm_dynamic_max_tokens,
                    minimum_max_tokens=vllm_minimum_max_tokens,
                    fail_on_request_error=vllm_fail_on_request_error,
```

- [ ] **Step 3: Add wrapper CLI/env flags**

In `2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py`, after the existing `--vllm-top-p` parser argument, add:

```python
    parser.add_argument("--vllm-total-max-length", type=int, default=int(os.environ.get("DTGPT_VLLM_TOTAL_MAX_LENGTH", "4092")))
    parser.add_argument("--vllm-dynamic-max-tokens", action="store_true", default=os.environ.get("DTGPT_VLLM_DYNAMIC_MAX_TOKENS", "1") in ["1", "true", "True", "yes", "YES"])
    parser.add_argument("--vllm-minimum-max-tokens", type=int, default=int(os.environ.get("DTGPT_VLLM_MINIMUM_MAX_TOKENS", "1")))
    parser.add_argument("--vllm-continue-on-request-error", action="store_true", default=os.environ.get("DTGPT_VLLM_CONTINUE_ON_REQUEST_ERROR", "0") in ["1", "true", "True", "yes", "YES"])
```

- [ ] **Step 4: Pass wrapper args into `experiment.run(...)`**

In the same wrapper, after:

```python
        vllm_top_p=args.vllm_top_p,
```

add:

```python
        vllm_total_max_length=args.vllm_total_max_length,
        vllm_dynamic_max_tokens=args.vllm_dynamic_max_tokens,
        vllm_minimum_max_tokens=args.vllm_minimum_max_tokens,
        vllm_fail_on_request_error=not args.vllm_continue_on_request_error,
```

- [ ] **Step 5: Run tests and syntax checks**

Run:

```bash
python -m unittest tests.test_vllm_generation_budget -v
python -m compileall pipeline 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row
```

Expected:

```text
Ran 6 tests
OK
```

and compileall returns success without `SyntaxError`.

- [ ] **Step 6: Commit MIMIC wiring**

```bash
git add pipeline/Experiment.py tests/test_vllm_generation_budget.py \
  1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py \
  1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py
git commit -m "Enable dynamic vLLM budgets for MIMIC eval

Constraint: MIMIC prompts can approach the BioMistral 4096-token serving limit.
Rejected: Reduce seq-max-len from 3400 globally | it removes clinical history even when output budget can be safely reduced instead.
Confidence: high
Scope-risk: moderate
Directive: Keep DTGPT_VLLM_DYNAMIC_MAX_TOKENS enabled for MIMIC vLLM jobs unless a larger-context model is served.
Tested: python -m unittest tests.test_vllm_generation_budget -v; python -m compileall pipeline 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row
Not-tested: full live MIMIC vLLM evaluation against checkpoint"
```

### Task 5: Smoke-test the wrapper configuration without launching a full evaluation

**Files:**
- Modify: none
- Test: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py`

- [ ] **Step 1: Check wrapper help includes new flags**

Run:

```bash
python 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py --help | grep -E "vllm-(total|max|minimum|continue)"
```

Expected output contains these flags:

```text
--vllm-total-max-length
--vllm-dynamic-max-tokens
--vllm-minimum-max-tokens
--vllm-continue-on-request-error
```

- [ ] **Step 2: Run final lightweight validation**

Run:

```bash
python -m unittest tests.test_vllm_generation_budget -v
python -m compileall pipeline 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row tests
```

Expected:

```text
Ran 6 tests
OK
```

and compileall returns success without `SyntaxError`.

- [ ] **Step 3: Commit smoke-test evidence if any tracked files changed**

If no files changed during smoke testing, do not create a commit. If a small tracked documentation note is added later, use:

```bash
git add <changed-file>
git commit -m "Document vLLM budget validation evidence

Constraint: Full MIMIC vLLM evaluation requires local checkpoint and running vLLM server.
Confidence: high
Scope-risk: narrow
Tested: python -m unittest tests.test_vllm_generation_budget -v; python -m compileall pipeline 1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row tests
Not-tested: full live MIMIC vLLM evaluation against checkpoint"
```

## Self-review

- **Spec coverage:** The log-end failure is covered by `test_dynamic_budget_matches_failed_36949_context_case`; the code path that sends fixed `max_tokens=900` is covered by Tasks 2-4; validation commands are listed in Tasks 1-5.
- **Placeholder scan:** No task uses unspecified code, unspecified paths, or deferred implementation text.
- **Type consistency:** The helper signature `compute_vllm_max_tokens(prompt, tokenizer, requested_max_new_tokens, total_max_length, dynamic_max_tokens, minimum_max_tokens)` is used consistently in tests and in `Experiment.get_output_for_split_vllm_completions(...)`.
