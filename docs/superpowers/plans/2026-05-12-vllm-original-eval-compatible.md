# vLLM Original Eval Compatible Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the vLLM evaluation path reproduce the original DT-GPT repository's MIMIC test behavior as closely as possible while still serving the DoRA/LoRA checkpoint through vLLM.

**Architecture:** Keep the existing DoRA/vLLM serving path, but change the vLLM client to use an original-repo-compatible generation budget: many stochastic samples, top-p sampling, mean aggregation, and per-request dynamic `max_tokens` derived from a HF-style total `max_length` budget. Add tests around token-budget calculation and request-failure handling so evaluation does not silently regress into fixed-output-token or single-sample behavior.

**Tech Stack:** Python 3.11, `unittest`, Hugging Face tokenizer interface, stdlib `urllib.request`, vLLM OpenAI-compatible `/v1/completions`, Slurm `sbatch`, existing DT-GPT `pipeline/Experiment.py` and MIMIC experiment scripts.

---

## Context and locked decisions

Original upstream repository evaluated MIMIC BioMistral with these settings in `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py`:

```python
batch_size_validation=5
seq_max_len_in_tokens=3400
num_samples_to_generate=30
sample_merging_strategy="mean"
max_new_tokens_to_generate=900
```

Original upstream `dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py` then called HF generation as:

```python
eval_targets, eval_prediction, return_meta_data_list = experiment.get_output_for_split_hf_default(
    eval_set_events,
    eval_manager,
    preprocessing_function=preprocessing_function,
    tokenizer=dp.tokenizer,
    encoding_function=encoding_function,
    decoding_function=decoding_function,
    post_processing_function=post_processing_function,
    batch_size=batch_size,
    verbose=verbose,
    gen_top_p=0.9,
    gen_do_sample=True,
    max_new_tokens=None,
    num_samples_to_generate=num_samples_to_generate,
    sample_merging_strategy=sample_merging_strategy,
    pad_token_id=dp.tokenizer.eos_token_id,
    max_output_length=4000,
    return_meta_data=True,
    note_down_probabilities=True,
)
```

HF `generate()` used `max_length=max_output_length + 92`, so the effective total sequence budget was `4092` tokens. vLLM `/completions` instead expects `max_tokens` as output-only budget. Therefore the vLLM-compatible equivalent must compute per-request:

```python
prompt_tokens = len(tokenizer(prompt).input_ids)
max_tokens = min(requested_max_new_tokens, total_max_length - prompt_tokens)
```

For original-compatible mode in this repo, use these defaults:

```text
DTGPT_SEQ_MAX_LEN=3400
DTGPT_NUM_SAMPLES_TO_GENERATE=30
DTGPT_MAX_NEW_TOKENS=900
DTGPT_VLLM_TOTAL_MAX_LENGTH=4092
DTGPT_VLLM_DYNAMIC_MAX_TOKENS=1
DTGPT_VLLM_TEMPERATURE=1.0
DTGPT_VLLM_TOP_P=0.9
DTGPT_MAX_CONCURRENT_REQUESTS=8
```

Run a smoke evaluation first before full array:

```text
DTGPT_EVAL_MAX_SAMPLES=100
DTGPT_NUM_SAMPLES_TO_GENERATE=30
```

---

## File structure

Modify:

- `pipeline/Experiment.py`
  - Add `compute_vllm_max_tokens()` helper near `align_prediction_to_target_shape()`.
  - Extend `get_output_for_split_vllm_completions()` with tokenizer-aware dynamic output token budgeting.
  - Allow request failures to degrade to a bad sample instead of failing an entire shard when `fail_on_request_error=False`.

- `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
  - Add run parameters for vLLM total-length budgeting and request failure policy.
  - Pass `dp.tokenizer` into the vLLM evaluation function.

- `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py`
  - Add CLI flags for original-compatible vLLM budgeting.
  - Keep original-repo defaults visible: `seq=3400`, `samples=30`, `max_new=900`, `temperature=1.0`, `top_p=0.9`.

- `job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh`
  - Change default client settings to original-compatible values.
  - Add environment controls for smoke vs full runs.
  - Print original-compatible settings into logs.

Create:

- `tests/test_vllm_generation_budget.py`
  - Unit tests for dynamic max-token calculation and error handling defaults.

No separate docs file is required beyond this implementation plan.

---

### Task 1: Add vLLM dynamic generation-budget helper

**Files:**
- Modify: `pipeline/Experiment.py`
- Create: `tests/test_vllm_generation_budget.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vllm_generation_budget.py` with this complete content:

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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
bash -ic 'conda activate dtgpt-unsloth && MPLCONFIGDIR=/tmp/mpl python -m unittest tests/test_vllm_generation_budget.py -v'
```

Expected: FAIL with an import error similar to:

```text
ImportError: cannot import name 'compute_vllm_max_tokens' from 'pipeline.Experiment'
```

- [ ] **Step 3: Add the helper implementation**

In `pipeline/Experiment.py`, insert this helper immediately after `align_prediction_to_target_shape()`:

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

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
bash -ic 'conda activate dtgpt-unsloth && MPLCONFIGDIR=/tmp/mpl python -m unittest tests/test_vllm_generation_budget.py -v'
```

Expected:

```text
Ran 5 tests
OK
```

- [ ] **Step 5: Commit**

```bash
git add pipeline/Experiment.py tests/test_vllm_generation_budget.py
git commit -m "Add vLLM generation budget helper

Constraint: vLLM completions uses output-token max_tokens while upstream DT-GPT used HF total max_length.
Rejected: Keep fixed max_new_tokens | it caused context overflows and differs from original eval semantics.
Confidence: high
Scope-risk: narrow
Directive: Keep dynamic token budgeting tokenizer-based, not string-length-based.
Tested: python -m unittest tests/test_vllm_generation_budget.py -v
Not-tested: Live vLLM server request path."
```

---

### Task 2: Use dynamic max-tokens in vLLM requests

**Files:**
- Modify: `pipeline/Experiment.py`
- Test: `tests/test_vllm_generation_budget.py`

- [ ] **Step 1: Add tests for per-request payload calculation helper**

Append these tests to `VllmGenerationBudgetTests` in `tests/test_vllm_generation_budget.py`:

```python
    def test_dynamic_budget_matches_previous_context_overflow_case(self):
        tokenizer = FakeTokenizer(token_count=3329)

        max_tokens = compute_vllm_max_tokens(
            prompt="ignored by fake tokenizer",
            tokenizer=tokenizer,
            requested_max_new_tokens=900,
            total_max_length=4092,
            dynamic_max_tokens=True,
            minimum_max_tokens=1,
        )

        self.assertEqual(max_tokens, 763)
        self.assertLessEqual(3329 + max_tokens, 4092)

    def test_dynamic_budget_prevents_vllm_4096_overflow_with_safety_total(self):
        tokenizer = FakeTokenizer(token_count=3329)

        max_tokens = compute_vllm_max_tokens(
            prompt="ignored by fake tokenizer",
            tokenizer=tokenizer,
            requested_max_new_tokens=900,
            total_max_length=4092,
            dynamic_max_tokens=True,
            minimum_max_tokens=1,
        )

        self.assertLess(3329 + max_tokens, 4096)
```

- [ ] **Step 2: Run tests to verify they pass before integration**

Run:

```bash
bash -ic 'conda activate dtgpt-unsloth && MPLCONFIGDIR=/tmp/mpl python -m unittest tests/test_vllm_generation_budget.py -v'
```

Expected:

```text
Ran 7 tests
OK
```

- [ ] **Step 3: Extend `get_output_for_split_vllm_completions()` signature**

In `pipeline/Experiment.py`, find the `get_output_for_split_vllm_completions` signature. Replace the ending parameters:

```python
                                              prediction_url="http://127.0.0.1:18101/v1/",
                                              model_name=None,
                                              max_concurrent_requests=16,
                                              temperature=1.0,
                                              top_p=0.9,
                                              num_samples_to_generate=1,
```

with:

```python
                                              prediction_url="http://127.0.0.1:18101/v1/",
                                              model_name=None,
                                              max_concurrent_requests=16,
                                              temperature=1.0,
                                              top_p=0.9,
                                              tokenizer=None,
                                              total_max_length=4092,
                                              dynamic_max_tokens=False,
                                              minimum_max_tokens=1,
                                              fail_on_request_error=True,
                                              num_samples_to_generate=1,
```

- [ ] **Step 4: Store per-request max_tokens when building requests**

In `pipeline/Experiment.py`, inside the loop that appends to `requests`, replace:

```python
                    requests.append({
                        "patientid": patientid,
                        "patient_sample_index": patient_sample_index,
                        "prompt": str_input,
                        "seed": 8719 + (idx * max(num_samples_to_generate, 1)) + sample_idx,
                    })
```

with:

```python
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

- [ ] **Step 5: Send per-request max_tokens to vLLM**

In `pipeline/Experiment.py`, inside `post_completion` payload construction, replace:

```python
                                "max_tokens": max_new_tokens,
```

with:

```python
                                "max_tokens": request["max_tokens"],
```

- [ ] **Step 6: Add useful budget logging**

In `pipeline/Experiment.py`, after all requests have been built and before `async def generate_all():`, add:

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

- [ ] **Step 7: Allow original-compatible vLLM to continue after request errors**

In `pipeline/Experiment.py`, replace:

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

- [ ] **Step 8: Run unit tests**

Run:

```bash
bash -ic 'conda activate dtgpt-unsloth && MPLCONFIGDIR=/tmp/mpl python -m unittest tests/test_vllm_generation_budget.py tests/test_prediction_alignment.py tests/test_metric_manager.py tests/test_evaluation_shards.py -v'
```

Expected:

```text
Ran 14 tests
OK
```

- [ ] **Step 9: Compile changed modules**

Run:

```bash
python -m py_compile pipeline/Experiment.py
```

Expected: no output and exit code `0`.

- [ ] **Step 10: Commit**

```bash
git add pipeline/Experiment.py tests/test_vllm_generation_budget.py
git commit -m "Match HF-style generation budget for vLLM eval

Constraint: Original DT-GPT used HF total max_length while vLLM requires output-token max_tokens.
Rejected: Increase vLLM max_model_len above 4096 | the served Mistral sliding-window limit rejected 8192 and the original eval used about 4092 total length.
Confidence: high
Scope-risk: moderate
Directive: Preserve per-request max_tokens; do not revert to one fixed max_new_tokens for vLLM eval.
Tested: python -m unittest tests/test_vllm_generation_budget.py tests/test_prediction_alignment.py tests/test_metric_manager.py tests/test_evaluation_shards.py -v; python -m py_compile pipeline/Experiment.py
Not-tested: Live vLLM server generation."
```

---

### Task 3: Thread original-compatible vLLM options through MIMIC run and CLI

**Files:**
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`
- Modify: `1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py`

- [ ] **Step 1: Extend the MIMIC run signature**

In `dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`, replace the final run parameters:

```python
                prediction_url="http://127.0.0.1:18101/v1/",
                vllm_model_name=None,
                max_concurrent_requests=16,
                vllm_temperature=1.0,
                vllm_top_p=0.9):
```

with:

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

- [ ] **Step 2: Pass tokenizer and budgeting options to vLLM evaluation**

In `dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py`, inside the `experiment.get_output_for_split_vllm_completions(...)` call, replace:

```python
                    max_concurrent_requests=max_concurrent_requests,
                    temperature=vllm_temperature,
                    top_p=vllm_top_p,
                )
```

with:

```python
                    max_concurrent_requests=max_concurrent_requests,
                    temperature=vllm_temperature,
                    top_p=vllm_top_p,
                    tokenizer=dp.tokenizer,
                    total_max_length=vllm_total_max_length,
                    dynamic_max_tokens=vllm_dynamic_max_tokens,
                    minimum_max_tokens=vllm_minimum_max_tokens,
                    fail_on_request_error=vllm_fail_on_request_error,
                )
```

- [ ] **Step 3: Add CLI arguments**

In `2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py`, inside `build_parser()`, after the existing `--vllm-top-p` line, add:

```python
    parser.add_argument("--vllm-total-max-length", type=int, default=int(os.environ.get("DTGPT_VLLM_TOTAL_MAX_LENGTH", "4092")))
    parser.add_argument("--vllm-dynamic-max-tokens", action="store_true", default=os.environ.get("DTGPT_VLLM_DYNAMIC_MAX_TOKENS", "0") in ["1", "true", "True", "yes", "YES"])
    parser.add_argument("--vllm-minimum-max-tokens", type=int, default=int(os.environ.get("DTGPT_VLLM_MINIMUM_MAX_TOKENS", "1")))
    parser.add_argument("--vllm-continue-on-request-error", action="store_true", default=os.environ.get("DTGPT_VLLM_CONTINUE_ON_REQUEST_ERROR", "0") in ["1", "true", "True", "yes", "YES"])
```

- [ ] **Step 4: Pass CLI arguments into `experiment.run()`**

In `2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py`, inside `experiment.run(...)`, after:

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

- [ ] **Step 5: Compile the experiment files**

Run:

```bash
python -m py_compile \
  1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py \
  1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py
```

Expected: no output and exit code `0`.

- [ ] **Step 6: Run unit tests**

Run:

```bash
bash -ic 'conda activate dtgpt-unsloth && MPLCONFIGDIR=/tmp/mpl python -m unittest tests/test_vllm_generation_budget.py tests/test_prediction_alignment.py tests/test_metric_manager.py tests/test_evaluation_shards.py -v'
```

Expected:

```text
Ran 14 tests
OK
```

- [ ] **Step 7: Commit**

```bash
git add \
  1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py \
  1_experiments/2024_02_08_mimic_iv/4_dt_gpt_instruction/2024_04_11_biomistral_td_bd_summarized_row/2024_04_15_dt_gpt_bd_bm_summarized_row_mimic_eval.py

git commit -m "Expose original-compatible vLLM eval options

Constraint: MIMIC eval scripts need to run both strict vLLM smoke tests and original-compatible stochastic multi-sample evaluation.
Rejected: Hardcode original-compatible behavior only in Experiment.py | CLI and Slurm need explicit reproducible knobs.
Confidence: high
Scope-risk: narrow
Directive: Keep original-compatible flags visible in logs and CLI rather than hidden defaults.
Tested: python -m py_compile MIMIC eval scripts; python -m unittest tests/test_vllm_generation_budget.py tests/test_prediction_alignment.py tests/test_metric_manager.py tests/test_evaluation_shards.py -v
Not-tested: Slurm submission."
```

---

### Task 4: Update Slurm job defaults to original-compatible vLLM evaluation

**Files:**
- Modify: `job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh`

- [ ] **Step 1: Change defaults and add logging variables**

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

- [ ] **Step 2: Print the new settings in the job log**

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

- [ ] **Step 3: Pass the new settings to the eval script**

In the eval command, replace:

```bash
    --num-samples-to-generate "${DTGPT_NUM_SAMPLES_TO_GENERATE:-1}" \
```

with:

```bash
    --num-samples-to-generate "${NUM_SAMPLES_TO_GENERATE}" \
```

Then replace the ending flags:

```bash
    --max-concurrent-requests "${MAX_CONCURRENT_REQUESTS}" \
    --vllm-temperature "${DTGPT_VLLM_TEMPERATURE:-1.0}" \
    --vllm-top-p "${DTGPT_VLLM_TOP_P:-0.9}"
```

with:

```bash
    --max-concurrent-requests "${MAX_CONCURRENT_REQUESTS}" \
    --vllm-temperature "${DTGPT_VLLM_TEMPERATURE:-1.0}" \
    --vllm-top-p "${DTGPT_VLLM_TOP_P:-0.9}" \
    --vllm-total-max-length "${VLLM_TOTAL_MAX_LENGTH}" \
    --vllm-minimum-max-tokens "${VLLM_MINIMUM_MAX_TOKENS}" \
    ${VLLM_DYNAMIC_MAX_TOKENS:+--vllm-dynamic-max-tokens} \
    ${VLLM_CONTINUE_ON_REQUEST_ERROR:+--vllm-continue-on-request-error}
```

- [ ] **Step 4: Run shell syntax check**

Run:

```bash
bash -n job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh
```

Expected: no output and exit code `0`.

- [ ] **Step 5: Inspect the eval command expansion safely**

Run:

```bash
DTGPT_EVAL_MAX_SAMPLES=2 \
DTGPT_NUM_SAMPLES_TO_GENERATE=30 \
DTGPT_SEQ_MAX_LEN=3400 \
DTGPT_MAX_NEW_TOKENS=900 \
DTGPT_VLLM_DYNAMIC_MAX_TOKENS=1 \
DTGPT_VLLM_CONTINUE_ON_REQUEST_ERROR=1 \
bash -n job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh
```

Expected: no output and exit code `0`.

- [ ] **Step 6: Commit**

```bash
git add job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh
git commit -m "Use original-compatible defaults for vLLM eval jobs

Constraint: Original DT-GPT MIMIC eval used 30 stochastic samples, top-p 0.9, mean aggregation, and about 4092 total sequence length.
Rejected: Keep one-sample vLLM defaults | one noisy sample produced unstable negative R2 despite positive rank correlation.
Confidence: high
Scope-risk: narrow
Directive: Use DTGPT_EVAL_MAX_SAMPLES for smoke tests before full 8-shard arrays.
Tested: bash -n job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh
Not-tested: Live Slurm job."
```

---

### Task 5: Run original-compatible smoke evaluation

**Files:**
- No source changes.
- Runtime logs expected under `logs/mimic_dora_vllm700_shard_<jobid>_0.out` and `logs/mimic_dora_vllm_server_<jobid>_0.log`.

- [ ] **Step 1: Submit a 100-sample smoke job**

Run:

```bash
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

Expected output:

```text
Submitted batch job <jobid>
```

- [ ] **Step 2: Monitor queue**

Run, replacing `<jobid>`:

```bash
/usr/bin/bash -lc 'squeue -u "$USER" | egrep "JOBID|<jobid>"'
```

Expected while running:

```text
<jobid>_0 ... R ...
```

- [ ] **Step 3: Verify log settings**

Run, replacing `<jobid>`:

```bash
grep -E "Client sequence max length|Max new tokens|Num samples to generate|vLLM total max length|vLLM dynamic max tokens|vLLM continue on request error|vLLM max_tokens budget summary" logs/mimic_dora_vllm700_shard_<jobid>_0.out
```

Expected lines:

```text
Client sequence max length: 3400
Max new tokens: 900
Num samples to generate: 30
vLLM total max length: 4092
vLLM dynamic max tokens: 1
vLLM continue on request error: 1
vLLM max_tokens budget summary: min=<positive>, max=<positive>, dynamic=True, total_max_length=4092
```

- [ ] **Step 4: Verify no context overflow**

Run, replacing `<jobid>`:

```bash
grep -RHEi "BadRequest|context length|maximum context length|HTTP 400|vLLM generation failed" logs/mimic_dora_vllm700_shard_<jobid>_0.err logs/mimic_dora_vllm700_shard_<jobid>_0.out logs/mimic_dora_vllm_server_<jobid>_0.log || true
```

Expected: no output. If output appears, inspect whether `vLLM max_tokens budget summary` contains `min=1`; if yes, reduce `DTGPT_SEQ_MAX_LEN` to `3200` for the next smoke job.

- [ ] **Step 5: Extract smoke metrics**

Run, replacing `<jobid>`:

```bash
grep -A10 "Resulting performances" logs/mimic_dora_vllm700_shard_<jobid>_0.out
```

Expected output includes:

```text
TEST all numeric columns r2 overall
TEST all numeric columns spearman corr overall
```

Acceptance threshold for proceeding to full array:

```text
No vLLM HTTP 400/context errors.
Prediction dataframe rows equal target dataframe rows.
Spearman is not worse than the previous one-sample run by more than 0.03.
R2 is better than or comparable to the one-sample run.
```

- [ ] **Step 6: Commit smoke evidence as a note if source changed after previous commits**

If source files changed during smoke debugging, commit them with a message following the Lore protocol. If no source files changed, do not commit logs.

---

### Task 6: Run full 8-shard original-compatible evaluation

**Files:**
- No source changes.
- Runtime logs expected under `logs/mimic_dora_vllm700_shard_<jobid>_<shard>.out`.

- [ ] **Step 1: Submit the full array only after smoke passes**

Run:

```bash
DTGPT_SEQ_MAX_LEN=3400 \
DTGPT_MAX_NEW_TOKENS=900 \
DTGPT_NUM_SAMPLES_TO_GENERATE=30 \
DTGPT_VLLM_TOTAL_MAX_LENGTH=4092 \
DTGPT_VLLM_DYNAMIC_MAX_TOKENS=1 \
DTGPT_VLLM_CONTINUE_ON_REQUEST_ERROR=1 \
DTGPT_MAX_CONCURRENT_REQUESTS=8 \
sbatch job/submit_mimic_dora_checkpoint700_vllm_eval_array.sh
```

Expected output:

```text
Submitted batch job <jobid>
```

- [ ] **Step 2: Monitor completion**

Run, replacing `<jobid>`:

```bash
/usr/bin/bash -lc 'squeue -u "$USER" | egrep "JOBID|<jobid>"'
```

Expected after completion: no `<jobid>` rows.

- [ ] **Step 3: Confirm every shard wrote result dataframes**

Run, replacing `<jobid>`:

```bash
for shard in 0 1 2 3 4 5 6 7; do
  echo "### shard ${shard}"
  grep -E "Saved target dataframe|Saved prediction dataframe|Resulting performances" logs/mimic_dora_vllm700_shard_<jobid>_${shard}.out
 done
```

Expected: each shard prints two saved dataframe lines plus one `Resulting performances` line.

- [ ] **Step 4: Confirm no shard has context errors**

Run, replacing `<jobid>`:

```bash
grep -RHEi "BadRequest|context length|maximum context length|HTTP 400|vLLM generation failed" \
  logs/mimic_dora_vllm700_shard_<jobid>_*.err \
  logs/mimic_dora_vllm700_shard_<jobid>_*.out \
  logs/mimic_dora_vllm_server_<jobid>_*.log || true
```

Expected: no output.

- [ ] **Step 5: Summarize metrics across shards**

Run this Python script from repo root, replacing `<jobid>` with the job id only to label output:

```bash
python - <<'PY'
import glob
import json
import os

paths = sorted(glob.glob('3_results/raw_experiments/DT-GPT/setup/setup/2026_*/eval_meta_data/TEST_shard_*_resulting_performances.json'))
rows = []
for path in paths:
    with open(path) as f:
        data = json.load(f)
    key = next(iter(data))
    if not key.startswith('TEST_shard_'):
        continue
    metrics = data[key]['all_numeric_columns']
    rows.append((
        key,
        metrics['r2']['overall'],
        metrics['mae']['overall'],
        metrics['rmse']['overall'],
        metrics['spearman_corr']['overall'],
        metrics['nrmse']['overall'],
        metrics['dir_accuracy']['overall'],
        os.path.dirname(path),
    ))

for row in rows[-8:]:
    print(row[0], 'r2=', round(row[1], 6), 'mae=', round(row[2], 6), 'rmse=', round(row[3], 6), 'spearman=', round(row[4], 6), 'nrmse=', round(row[5], 6), 'dir=', round(row[6], 6), row[7])
PY
```

Expected: eight shard rows for the latest full array.

- [ ] **Step 6: Decide whether to continue fine-tuning**

Use this decision rule:

```text
If 30-sample original-compatible vLLM improves R2 meaningfully versus the one-sample run, keep checkpoint-700 and report test results.
If Spearman remains around 0.35-0.40 but R2 remains negative, consider calibration/post-processing before more fine-tuning.
If both Spearman and R2 stay poor, continue fine-tuning from checkpoint-700 instead of restarting from base.
```

Do not submit further training in this task unless the user explicitly asks for fine-tuning after reviewing the corrected evaluation results.

---

## Self-review

**Spec coverage:**
- Restores original repo test parameters through vLLM: Task 4 and Task 6.
- Preserves DoRA/vLLM backend instead of requiring full HF inference: Task 2 and Task 3.
- Handles original HF `max_length` vs vLLM `max_tokens`: Task 1 and Task 2.
- Prevents shard loss from single request overflow: Task 2 and Task 5.
- Provides smoke-before-full workflow: Task 5 and Task 6.

**Placeholder scan:**
- No `TBD`, `TODO`, or undefined implementation placeholders remain.
- Commands include expected outputs.
- Code snippets define exact functions, args, and replacements.

**Type consistency:**
- `compute_vllm_max_tokens()` uses `tokenizer(prompt, add_special_tokens=False).input_ids` consistently with HF tokenizer behavior.
- CLI names use `vllm_total_max_length`, `vllm_dynamic_max_tokens`, `vllm_minimum_max_tokens`, and `vllm_continue_on_request_error`; run-function names match except the run receives `vllm_fail_on_request_error=not args.vllm_continue_on_request_error`.
- Slurm env names map to CLI flags exactly.
