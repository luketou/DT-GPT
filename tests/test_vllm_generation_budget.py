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
