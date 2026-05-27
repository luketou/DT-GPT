import unittest

from pipeline.data_processors.DataProcessorBiomistral import CompletionOnlyDataCollator


class FakeTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, text, add_special_tokens=False):
        if text != "<patient_prediction>":
            raise AssertionError(f"unexpected tokenized text: {text!r}")
        return {"input_ids": [101, 102]}


class CompletionOnlyDataCollatorTest(unittest.TestCase):
    def test_masks_prompt_template_and_padding_labels(self):
        collator = CompletionOnlyDataCollator("<patient_prediction>", tokenizer=FakeTokenizer())

        batch = collator([
            {"input_ids": [10, 11, 101, 102, 20, 21]},
            {"input_ids": [101, 102, 30]},
        ])

        self.assertEqual(batch["input_ids"].tolist(), [
            [10, 11, 101, 102, 20, 21],
            [101, 102, 30, 0, 0, 0],
        ])
        self.assertEqual(batch["attention_mask"].tolist(), [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ])
        self.assertEqual(batch["labels"].tolist(), [
            [-100, -100, -100, -100, 20, 21],
            [-100, -100, 30, -100, -100, -100],
        ])

    def test_masks_everything_when_template_is_missing(self):
        collator = CompletionOnlyDataCollator("<patient_prediction>", tokenizer=FakeTokenizer())

        batch = collator([{"input_ids": [10, 11, 20]}])

        self.assertEqual(batch["labels"].tolist(), [[-100, -100, -100]])


if __name__ == "__main__":
    unittest.main()
