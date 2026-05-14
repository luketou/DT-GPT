# vLLM Dynamic Budget Failure Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the failed `mimic_dora_vllm700` evaluation array by making vLLM requests use tokenizer-aware per-prompt `max_tokens` and by passing original-compatible vLLM controls through CLI and Slurm.

**Architecture:** Keep the current merged-model vLLM serving path in `job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh`, but change the client request builder in `pipeline/Experiment.py` to compute output-token budgets dynamically from a Hugging Face tokenizer and a total context budget. Thread the tokenizer and knobs from the MIMIC run script, CLI, and Slurm job, then run a 100-sample smoke job before a full 8-shard array.

**Tech Stack:** Python 3.11, stdlib `unittest`, stdlib `urllib.request`, Hugging Face tokenizer interface, vLLM OpenAI-compatible `/v1/completions`, Slurm `sbatch`, existing DT-GPT MIMIC eval scripts.

---

## Observed failure evidence

The submitted array was job `36949` with shards `0..7`:

```text
logs/mimic_dora_vllm700_shard_36949_0.out ... logs/mimic_dora_vllm700_shard_36949_7.out
logs/mimic_dora_vllm_server_36949_0.log ... logs/mimic_dora_vllm_server_36949_7.log
```

Every shard started vLLM successfully:

```text
vLLM server is ready after 11-16 checks.
```

Every shard then failed during client generation. Representative client error from `logs/mimic_dora_vllm700_shard_36949_0.err`:

```text
RuntimeError: vLLM generation failed for 7380 request(s); first failed request indices: [2, 1, 3, 4, 5, 7, 6, 8, 10, 9, 11, 13, 14, 16, 12, 17, 15, 19, 18, 20]
```

Representative server error from `logs/mimic_dora_vllm_server_36949_0.log`:

```text
vllm.exceptions.VLLMValidationError: This model's maximum context length is 4096 tokens. However, you requested 900 output tokens and your prompt contains at least 3197 input tokens, for a total of at least 4097 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=3197)
```

## Root cause

`pipeline/Experiment.py:get_output_for_split_vllm_completions()` sends a fixed output budget for every request:

```python
"max_tokens": max_new_tokens,
```

The full array was submitted with original-compatible generation values, including 900 requested output tokens. For prompts with 3197 tokens, vLLM rejects `3197 + 900 = 4097`, which is one token over the 4096 model context limit. The code has no dynamic HF-style total-length conversion, no CLI/Slurm flags for that conversion, and raises the whole shard when any request fails.

## File structure

Modify:

- `pipeline/Experiment.py`
  - Add `compute_vllm_max_tokens()` near `align_prediction_to_target_shape()`.
  - Extend `get_output_for_split_vllm_completions()` with tokenizer-aware dynamic budgeting.
  - Store `max_tokens` per request and send it in the vLLM payload.
  - Add concise budget/error logging.
  - Add optional `fail_on_request_error` policy so smoke jobs can continue with bad samples when explicitly requested.

- `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
  - Add vLLM budget/run parameters to `run()`.
  - Pass `dp.tokenizer` and budgeting flags into the vLLM eval call.

- `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py`
  - Add CLI/env flags for total length, dynamic max tokens, minimum max tokens, and continue-on-request-error.
  - Pass these flags into `experiment.run()`.

- `job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh`
  - Change defaults to original-compatible values.
  - Pass dynamic-budget flags into the eval script.
  - Print the new settings into logs.

Create:

- `tests/test_vllm_generation_budget.py`
  - Unit tests for dynamic token budget calculation.

---

### Task 1: Add failing tests for vLLM max-token budgeting

**Files:**
- Create: `tests/test_vllm_generation_budget.py`

- [ ] **Step 1: Create the test file**

Create `tests/test_vllm_generation_budget.py` with exactly this content:

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

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
bash -ic 'conda activate dtgpt-unsloth && MPLCONFIGDIR=/tmp/mpl python -m unittest tests/test_vllm_generation_budget.py -v'
```

Expected: fail with import error:

```text
ImportError: cannot import name 'compute_vllm_max_tokens' from 'pipeline.Experiment'
```

---

### Task 2: Implement the dynamic budget helper

**Files:**
- Modify: `pipeline/Experiment.py`
- Test: `tests/test_vllm_generation_budget.py`

- [ ] **Step 1: Add helper after `align_prediction_to_target_shape()`**

In `pipeline/Experiment.py`, find `align_prediction_to_target_shape()`. Immediately after that function, add:

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

- [ ] **Step 2: Run helper tests**

Run:

```bash
bash -ic 'conda activate dtgpt-unsloth && MPLCONFIGDIR=/tmp/mpl python -m unittest tests/test_vllm_generation_budget.py -v'
```

Expected:

```text
Ran 6 tests
OK
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/Experiment.py tests/test_vllm_generation_budget.py
git commit -m "Add vLLM dynamic token budget helper

Constraint: vLLM rejects requests when prompt_tokens + max_tokens exceeds the served model context length.
Rejected: Keep fixed max_new_tokens | job 36949 failed because 3197 prompt tokens plus 900 output tokens exceeded 4096.
Confidence: high
Scope-risk: narrow
Directive: Keep dynamic budgeting tokenizer-based and per-request.
Tested: python -m unittest tests/test_vllm_generation_budget.py -v
Not-tested: Live vLLM request path."
```

---

### Task 3: Use per-request `max_tokens` in vLLM completions

**Files:**
- Modify: `pipeline/Experiment.py`
- Test: `tests/test_vllm_generation_budget.py`

- [ ] **Step 1: Extend `get_output_for_split_vllm_completions()` signature**

In `pipeline/Experiment.py`, find:

```python
    def get_output_for_split_vllm_completions(self, list_of_split_dfs, eval_manager, preprocessing_function,
                                              post_processing_function, max_output_length,
                                              encoding_function,
                                              prediction_url,
                                              model_name,
                                              max_concurrent_requests=16,
                                              temperature=1.0,
                                              top_p=0.9,
                                              num_samples_to_generate=1,
                                              sample_merging_strategy="mean",
                                              max_new_tokens=None,
                                              output_string_filtering_function=None,
                                              return_meta_data=False,
                                              verbose=False):
```

Replace it with:

```python
    def get_output_for_split_vllm_completions(self, list_of_split_dfs, eval_manager, preprocessing_function,
                                              post_processing_function, max_output_length,
                                              encoding_function,
                                              prediction_url,
                                              model_name,
                                              max_concurrent_requests=16,
                                              temperature=1.0,
                                              top_p=0.9,
                                              tokenizer=None,
                                              total_max_length=4092,
                                              dynamic_max_tokens=False,
                                              minimum_max_tokens=1,
                                              fail_on_request_error=True,
                                              num_samples_to_generate=1,
                                              sample_merging_strategy="mean",
                                              max_new_tokens=None,
                                              output_string_filtering_function=None,
                                              return_meta_data=False,
                                              verbose=False):
```

- [ ] **Step 2: Store `max_tokens` in each request**

Inside the request-building loop, replace:

```python
                for sample_idx in range(num_samples_to_generate):
                    requests.append({
                        "patientid": patientid,
                        "patient_sample_index": patient_sample_index,
                        "prompt": str_input,
                        "seed": 8719 + (idx * max(num_samples_to_generate, 1)) + sample_idx,
                    })
```

with:

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

- [ ] **Step 3: Add budget logging before `generate_all()`**

Immediately before:

```python
        async def generate_all():
```

add:

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

- [ ] **Step 4: Send per-request `max_tokens`**

Inside the payload passed to `post_completion`, replace:

```python
                                "max_tokens": max_new_tokens,
```

with:

```python
                                "max_tokens": request["max_tokens"],
```

- [ ] **Step 5: Preserve HTTP error details in log output**

Replace the broad exception handler inside `generate_one()`:

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

- [ ] **Step 6: Make failed-request policy configurable**

Replace:

```python
            if failed_requests:
                raise RuntimeError(
                    "vLLM generation failed for "
                    + str(len(failed_requests))
                    + " request(s); first failed request indices: "
                    + str(failed_requests[:20])
                )
            return results
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
            return results
```

- [ ] **Step 7: Run unit and syntax tests**

Run:

```bash
bash -ic 'conda activate dtgpt-unsloth && MPLCONFIGDIR=/tmp/mpl python -m unittest tests/test_vllm_generation_budget.py tests/test_prediction_alignment.py tests/test_metric_manager.py tests/test_evaluation_shards.py -v'
python -m py_compile pipeline/Experiment.py
```

Expected:

```text
OK
```

and `py_compile` exits with code `0` and no output.

- [ ] **Step 8: Commit**

```bash
git add pipeline/Experiment.py tests/test_vllm_generation_budget.py
git commit -m "Use dynamic max_tokens for vLLM completions

Constraint: Job 36949 failed because the vLLM client sent fixed 900-token outputs for prompts near the 4096-token context limit.
Rejected: Only reduce global max_new_tokens | it would waste valid output budget for shorter prompts and diverge from HF total-length semantics.
Confidence: high
Scope-risk: moderate
Directive: Preserve request-level max_tokens and budget-summary logging.
Tested: python -m unittest tests/test_vllm_generation_budget.py tests/test_prediction_alignment.py tests/test_metric_manager.py tests/test_evaluation_shards.py -v; python -m py_compile pipeline/Experiment.py
Not-tested: Live vLLM server generation."
```

---

### Task 4: Thread dynamic-budget options through the MIMIC run and CLI

**Files:**
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py`

- [ ] **Step 1: Extend `run()` signature**

In `dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`, find the final vLLM parameters in `run()`:

```python
                prediction_url="http://127.0.0.1:18101/v1/",
                vllm_model_name=None,
                max_concurrent_requests=16,
                vllm_temperature=1.0,
                vllm_top_p=0.9):
```

Replace with:

```python
                prediction_url="http://127.0.0.1:18101/v1/",
                vllm_model_name=None,
                max_concurrent_requests=16,
                vllm_temperature=1.0,
                vllm_top_p=0.9,
                vllm_total_max_length=4092,
                vllm_dynamic_max_tokens=False,
                vllm_minimum_max_tokens=1,
                vllm_fail_on_request_error=True):
```

- [ ] **Step 2: Pass tokenizer and dynamic-budget options to vLLM eval**

In `dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`, inside `experiment.get_output_for_split_vllm_completions(...)`, after:

```python
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

- [ ] **Step 3: Add CLI arguments**

In `2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py`, inside `build_parser()`, after:

```python
    parser.add_argument("--vllm-top-p", type=float, default=float(os.environ.get("DTGPT_VLLM_TOP_P", "0.9")))
```

add:

```python
    parser.add_argument("--vllm-total-max-length", type=int, default=int(os.environ.get("DTGPT_VLLM_TOTAL_MAX_LENGTH", "4092")))
    parser.add_argument("--vllm-dynamic-max-tokens", action="store_true", default=os.environ.get("DTGPT_VLLM_DYNAMIC_MAX_TOKENS", "0") in ["1", "true", "True", "yes", "YES"])
    parser.add_argument("--vllm-minimum-max-tokens", type=int, default=int(os.environ.get("DTGPT_VLLM_MINIMUM_MAX_TOKENS", "1")))
    parser.add_argument("--vllm-continue-on-request-error", action="store_true", default=os.environ.get("DTGPT_VLLM_CONTINUE_ON_REQUEST_ERROR", "0") in ["1", "true", "True", "yes", "YES"])
```

- [ ] **Step 4: Pass CLI arguments into `experiment.run()`**

In the same file, inside `experiment.run(...)`, after:

```python
        vllm_temperature=args.vllm_temperature,
        vllm_top_p=args.vllm_top_p,
```

add:

```python
        vllm_total_max_length=args.vllm_total_max_length,
        vllm_dynamic_max_tokens=args.vllm_dynamic_max_tokens,
        vllm_minimum_max_tokens=args.vllm_minimum_max_tokens,
        vllm_fail_on_request_error=not args.vllm_continue_on_request_error,
```

- [ ] **Step 5: Compile scripts**

Run:

```bash
python -m py_compile \
  1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py \
  1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py
```

Expected: no output and exit code `0`.

- [ ] **Step 6: Commit**

```bash
git add \
  1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py \
  1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py

git commit -m "Expose dynamic vLLM budget controls

Constraint: The MIMIC eval CLI and Slurm job must reproduce original HF-style total-length budgeting through vLLM.
Rejected: Hidden hardcoded dynamic mode | smoke and full jobs need visible reproducible knobs.
Confidence: high
Scope-risk: narrow
Directive: Keep tokenizer, total_max_length, dynamic_max_tokens, minimum_max_tokens, and request-error policy wired together.
Tested: python -m py_compile MIMIC eval scripts
Not-tested: Live Slurm job."
```

---

### Task 5: Update Slurm job defaults and flags

**Files:**
- Modify: `job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh`

- [ ] **Step 1: Change client defaults**

In `job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh`, replace:

```bash
CLIENT_SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-2900}"
MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-768}"
MAX_CONCURRENT_REQUESTS="${DTGPT_MAX_CONCURRENT_REQUESTS:-8}"
```

with:

```bash
CLIENT_SEQ_MAX_LEN="${DTGPT_SEQ_MAX_LEN:-3400}"
MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-900}"
MAX_CONCURRENT_REQUESTS="${DTGPT_MAX_CONCURRENT_REQUESTS:-8}"
NUM_SAMPLES_TO_GENERATE="${DTGPT_NUM_SAMPLES_TO_GENERATE:-30}"
VLLM_TOTAL_MAX_LENGTH="${DTGPT_VLLM_TOTAL_MAX_LENGTH:-4092}"
VLLM_DYNAMIC_MAX_TOKENS="${DTGPT_VLLM_DYNAMIC_MAX_TOKENS:-1}"
VLLM_MINIMUM_MAX_TOKENS="${DTGPT_VLLM_MINIMUM_MAX_TOKENS:-1}"
VLLM_CONTINUE_ON_REQUEST_ERROR="${DTGPT_VLLM_CONTINUE_ON_REQUEST_ERROR:-1}"
```

- [ ] **Step 2: Add explicit boolean flag arrays**

After the variable block from Step 1, add:

```bash
DYNAMIC_MAX_TOKENS_FLAG=()
CONTINUE_ON_REQUEST_ERROR_FLAG=()
if [[ "${VLLM_DYNAMIC_MAX_TOKENS}" == "1" || "${VLLM_DYNAMIC_MAX_TOKENS}" == "true" || "${VLLM_DYNAMIC_MAX_TOKENS}" == "True" || "${VLLM_DYNAMIC_MAX_TOKENS}" == "yes" || "${VLLM_DYNAMIC_MAX_TOKENS}" == "YES" ]]; then
    DYNAMIC_MAX_TOKENS_FLAG=(--vllm-dynamic-max-tokens)
fi
if [[ "${VLLM_CONTINUE_ON_REQUEST_ERROR}" == "1" || "${VLLM_CONTINUE_ON_REQUEST_ERROR}" == "true" || "${VLLM_CONTINUE_ON_REQUEST_ERROR}" == "True" || "${VLLM_CONTINUE_ON_REQUEST_ERROR}" == "yes" || "${VLLM_CONTINUE_ON_REQUEST_ERROR}" == "YES" ]]; then
    CONTINUE_ON_REQUEST_ERROR_FLAG=(--vllm-continue-on-request-error)
fi
```

- [ ] **Step 3: Print new settings**

After:

```bash
echo "Max concurrent requests: ${MAX_CONCURRENT_REQUESTS}"
```

add:

```bash
echo "Num samples to generate: ${NUM_SAMPLES_TO_GENERATE}"
echo "vLLM total max length: ${VLLM_TOTAL_MAX_LENGTH}"
echo "vLLM dynamic max tokens: ${VLLM_DYNAMIC_MAX_TOKENS}"
echo "vLLM minimum max tokens: ${VLLM_MINIMUM_MAX_TOKENS}"
echo "vLLM continue on request error: ${VLLM_CONTINUE_ON_REQUEST_ERROR}"
```

- [ ] **Step 4: Use `NUM_SAMPLES_TO_GENERATE` in eval command**

In the final eval command, replace:

```bash
    --num-samples-to-generate "${DTGPT_NUM_SAMPLES_TO_GENERATE:-1}" \
```

with:

```bash
    --num-samples-to-generate "${NUM_SAMPLES_TO_GENERATE}" \
```

- [ ] **Step 5: Pass dynamic-budget CLI flags**

In the final eval command, after:

```bash
    --vllm-top-p "${DTGPT_VLLM_TOP_P:-0.9}"
```

change the ending to:

```bash
    --vllm-top-p "${DTGPT_VLLM_TOP_P:-0.9}" \
    --vllm-total-max-length "${VLLM_TOTAL_MAX_LENGTH}" \
    --vllm-minimum-max-tokens "${VLLM_MINIMUM_MAX_TOKENS}" \
    "${DYNAMIC_MAX_TOKENS_FLAG[@]}" \
    "${CONTINUE_ON_REQUEST_ERROR_FLAG[@]}"
```

- [ ] **Step 6: Run shell syntax checks**

Run:

```bash
bash -n job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh
DTGPT_EVAL_MAX_SAMPLES=2 \
DTGPT_NUM_SAMPLES_TO_GENERATE=30 \
DTGPT_SEQ_MAX_LEN=3400 \
DTGPT_MAX_NEW_TOKENS=900 \
DTGPT_VLLM_DYNAMIC_MAX_TOKENS=1 \
DTGPT_VLLM_CONTINUE_ON_REQUEST_ERROR=1 \
bash -n job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh
```

Expected: no output and exit code `0` for both commands.

- [ ] **Step 7: Commit**

```bash
git add job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh
git commit -m "Pass dynamic vLLM budget flags from Slurm

Constraint: Job 36949 submitted original-compatible 900-token output requests but the client had no dynamic max_tokens controls.
Rejected: Rely on environment variables only | the job log and command line must prove which budget policy was used.
Confidence: high
Scope-risk: narrow
Directive: Smoke with DTGPT_EVAL_MAX_SAMPLES=100 before submitting the full array.
Tested: bash -n job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh
Not-tested: Live Slurm evaluation."
```

---

### Task 6: Smoke-test the fixed evaluation path

**Files:**
- No source changes expected.

- [ ] **Step 1: Submit a 100-sample smoke job**

Run from repo root:

```bash
DTGPT_EVAL_MODEL_PATH=/share/home/r15543056/trajectory_forecast/DT-GPT/3_results/raw_experiments/DT-GPTsetup/setup/2026_05_04___20_28_41_718957/models/checkpoint-700 \
DTGPT_EVAL_MAX_SAMPLES=100 \
DTGPT_EVAL_NUM_SHARDS=1 \
DTGPT_SEQ_MAX_LEN=3400 \
DTGPT_MAX_NEW_TOKENS=900 \
DTGPT_NUM_SAMPLES_TO_GENERATE=30 \
DTGPT_VLLM_TOTAL_MAX_LENGTH=4092 \
DTGPT_VLLM_DYNAMIC_MAX_TOKENS=1 \
DTGPT_VLLM_CONTINUE_ON_REQUEST_ERROR=1 \
DTGPT_MAX_CONCURRENT_REQUESTS=8 \
sbatch --array=0-0 job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh
```

Expected:

```text
Submitted batch job <jobid>
```

- [ ] **Step 2: Verify log settings**

Replace `<jobid>` with the submitted job id:

```bash
grep -E "Client sequence max length|Max new tokens|Num samples to generate|vLLM total max length|vLLM dynamic max tokens|vLLM continue on request error|vLLM max_tokens budget summary" logs/mimic_dora_vllm700_shard_<jobid>_0.out
```

Expected lines include:

```text
Client sequence max length: 3400
Max new tokens: 900
Num samples to generate: 30
vLLM total max length: 4092
vLLM dynamic max tokens: 1
vLLM continue on request error: 1
vLLM max_tokens budget summary: min=<positive>, max=900, dynamic=True, total_max_length=4092
```

- [ ] **Step 3: Verify no context-length failures**

Run:

```bash
grep -RHEi "BadRequest|context length|maximum context length|HTTP 400|vLLM generation failed" \
  logs/mimic_dora_vllm700_shard_<jobid>_0.err \
  logs/mimic_dora_vllm700_shard_<jobid>_0.out \
  logs/mimic_dora_vllm_server_<jobid>_0.log || true
```

Expected: no output.

- [ ] **Step 4: Verify outputs and metrics**

Run:

```bash
grep -E "Saved target dataframe|Saved prediction dataframe|Resulting performances" logs/mimic_dora_vllm700_shard_<jobid>_0.out
grep -A10 "Resulting performances" logs/mimic_dora_vllm700_shard_<jobid>_0.out
```

Expected: saved target/prediction dataframe lines and metric rows including:

```text
TEST all numeric columns r2 overall
TEST all numeric columns spearman corr overall
```

- [ ] **Step 5: Commit source changes from smoke debugging if any**

If smoke uncovered a source bug and code changed, commit with a Lore-format message. If only logs changed, do not commit logs.

---

### Task 7: Run the full 8-shard fixed array after smoke passes

**Files:**
- No source changes expected.

- [ ] **Step 1: Submit full array**

Only run this after Task 6 has no context-length failures:

```bash
DTGPT_EVAL_MODEL_PATH=/share/home/r15543056/trajectory_forecast/DT-GPT/3_results/raw_experiments/DT-GPTsetup/setup/2026_05_04___20_28_41_718957/models/checkpoint-700 \
DTGPT_SEQ_MAX_LEN=3400 \
DTGPT_MAX_NEW_TOKENS=900 \
DTGPT_NUM_SAMPLES_TO_GENERATE=30 \
DTGPT_VLLM_TOTAL_MAX_LENGTH=4092 \
DTGPT_VLLM_DYNAMIC_MAX_TOKENS=1 \
DTGPT_VLLM_CONTINUE_ON_REQUEST_ERROR=1 \
DTGPT_MAX_CONCURRENT_REQUESTS=8 \
sbatch job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh
```

Expected:

```text
Submitted batch job <jobid>
```

- [ ] **Step 2: Confirm no shard has context failures**

Run:

```bash
grep -RHEi "BadRequest|context length|maximum context length|HTTP 400|vLLM generation failed" \
  logs/mimic_dora_vllm700_shard_<jobid>_*.err \
  logs/mimic_dora_vllm700_shard_<jobid>_*.out \
  logs/mimic_dora_vllm_server_<jobid>_*.log || true
```

Expected: no output.

- [ ] **Step 3: Confirm every shard produced dataframes**

Run:

```bash
for shard in 0 1 2 3 4 5 6 7; do
  echo "### shard ${shard}"
  grep -E "Saved target dataframe|Saved prediction dataframe|Resulting performances" logs/mimic_dora_vllm700_shard_<jobid>_${shard}.out
 done
```

Expected: each shard prints saved target dataframe, saved prediction dataframe, and resulting performances.

---

## Self-review

**Spec coverage:** This plan addresses the observed job `36949` failures, fixes fixed `max_tokens=900`, adds tokenizer-aware per-request budgeting, exposes CLI/Slurm controls, preserves server startup behavior, and defines smoke-before-full verification.

**Placeholder scan:** No undefined implementation placeholders are present. All code edits include concrete snippets and commands include expected outputs.

**Type consistency:** Function names and parameter names are consistent across tasks: `compute_vllm_max_tokens`, `tokenizer`, `total_max_length`, `dynamic_max_tokens`, `minimum_max_tokens`, and `fail_on_request_error` are threaded from Slurm to CLI to run script to `Experiment.py`.
