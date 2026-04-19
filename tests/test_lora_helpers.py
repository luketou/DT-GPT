import unittest

from pipeline.lora_helpers import (
    DEFAULT_MISTRAL_LORA_TARGET_MODULES,
    build_lora_adapter_path,
    build_mistral_lora_config,
)


class LoraHelperTests(unittest.TestCase):
    def test_build_mistral_lora_config_uses_expected_defaults(self):
        config = build_mistral_lora_config()
        self.assertEqual(config.r, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertAlmostEqual(config.lora_dropout, 0.05)
        self.assertEqual(config.bias, "none")
        self.assertEqual(config.task_type.value, "CAUSAL_LM")
        self.assertEqual(
            tuple(config.target_modules),
            DEFAULT_MISTRAL_LORA_TARGET_MODULES,
        )

    def test_build_lora_adapter_path_uses_adapter_directory_name(self):
        self.assertEqual(
            build_lora_adapter_path("/tmp/experiment/"),
            "/tmp/experiment/fine_tuned_lora_adapter",
        )

    def test_build_mistral_lora_config_supports_overrides(self):
        config = build_mistral_lora_config(r=8, lora_alpha=16, lora_dropout=0.1)
        self.assertEqual(config.r, 8)
        self.assertEqual(config.lora_alpha, 16)
        self.assertAlmostEqual(config.lora_dropout, 0.1)


if __name__ == "__main__":
    unittest.main()
