import unittest

from pipeline.hf_training_args import normalize_training_argument_kwargs


class ModernTrainingArguments:
    def __init__(self, output_dir=None, eval_strategy="no", save_strategy="no"):
        self.output_dir = output_dir
        self.eval_strategy = eval_strategy
        self.save_strategy = save_strategy


class LegacyTrainingArguments:
    def __init__(self, output_dir=None, evaluation_strategy="no", save_strategy="no"):
        self.output_dir = output_dir
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy


class TrainingArgumentsCompatibilityTest(unittest.TestCase):
    def test_maps_evaluation_strategy_to_eval_strategy_for_modern_transformers(self):
        normalized = normalize_training_argument_kwargs(
            {
                "output_dir": "model-output",
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
            },
            ModernTrainingArguments,
        )

        self.assertEqual(
            normalized,
            {
                "output_dir": "model-output",
                "eval_strategy": "steps",
                "save_strategy": "steps",
            },
        )

    def test_keeps_evaluation_strategy_for_legacy_transformers(self):
        normalized = normalize_training_argument_kwargs(
            {
                "output_dir": "model-output",
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
            },
            LegacyTrainingArguments,
        )

        self.assertEqual(
            normalized,
            {
                "output_dir": "model-output",
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
            },
        )

    def test_maps_eval_strategy_to_evaluation_strategy_for_legacy_transformers(self):
        normalized = normalize_training_argument_kwargs(
            {
                "output_dir": "model-output",
                "eval_strategy": "steps",
                "save_strategy": "steps",
            },
            LegacyTrainingArguments,
        )

        self.assertEqual(
            normalized,
            {
                "output_dir": "model-output",
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
            },
        )

    def test_rejects_duplicate_eval_strategy_aliases(self):
        with self.assertRaisesRegex(ValueError, "Both 'evaluation_strategy' and 'eval_strategy'"):
            normalize_training_argument_kwargs(
                {
                    "output_dir": "model-output",
                    "evaluation_strategy": "steps",
                    "eval_strategy": "epoch",
                },
                ModernTrainingArguments,
            )


if __name__ == "__main__":
    unittest.main()
