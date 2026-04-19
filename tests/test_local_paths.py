import unittest

from pipeline.local_paths import (
    resolve_biomistral_model_path,
    resolve_tokenizer_model_path,
    select_precision_config,
)


class LocalPathTests(unittest.TestCase):
    def test_resolve_biomistral_model_path_prefers_env_override(self):
        env = {"DTGPT_BIOMISTRAL_MODEL_PATH": "/tmp/custom-biomistral"}
        resolved = resolve_biomistral_model_path(env=env, local_candidate_exists=False)
        self.assertEqual(resolved, "/tmp/custom-biomistral")

    def test_resolve_biomistral_model_path_falls_back_to_hf_name(self):
        resolved = resolve_biomistral_model_path(env={}, local_candidate_exists=False)
        self.assertEqual(resolved, "BioMistral/BioMistral-7B-DARE")

    def test_resolve_tokenizer_model_path_prefers_env_override(self):
        env = {"DTGPT_TOKENIZER_MODEL_PATH": "/tmp/custom-tokenizer"}
        resolved = resolve_tokenizer_model_path(env=env, local_candidate_exists=False)
        self.assertEqual(resolved, "/tmp/custom-tokenizer")

    def test_resolve_tokenizer_model_path_falls_back_to_biomistral(self):
        resolved = resolve_tokenizer_model_path(env={}, local_candidate_exists=False)
        self.assertEqual(resolved, "BioMistral/BioMistral-7B-DARE")

    def test_select_precision_config_uses_bfloat16_for_ampere(self):
        cfg = select_precision_config(
            cuda_available=True,
            capability_major=8,
            training=True,
        )
        self.assertEqual(cfg["torch_dtype_name"], "bfloat16")
        self.assertTrue(cfg["bf16"])
        self.assertFalse(cfg["fp16"])
        self.assertEqual(cfg["attn_implementation"], "flash_attention_2")

    def test_select_precision_config_uses_fp32_weights_for_v100_training(self):
        cfg = select_precision_config(
            cuda_available=True,
            capability_major=7,
            training=True,
        )
        self.assertEqual(cfg["torch_dtype_name"], "float32")
        self.assertFalse(cfg["bf16"])
        self.assertTrue(cfg["fp16"])
        self.assertEqual(cfg["attn_implementation"], "eager")

    def test_select_precision_config_uses_fp16_for_v100_inference(self):
        cfg = select_precision_config(
            cuda_available=True,
            capability_major=7,
            training=False,
        )
        self.assertEqual(cfg["torch_dtype_name"], "float16")
        self.assertFalse(cfg["bf16"])
        self.assertTrue(cfg["fp16"])
        self.assertEqual(cfg["attn_implementation"], "eager")

    def test_select_precision_config_uses_fp32_without_cuda(self):
        cfg = select_precision_config(
            cuda_available=False,
            capability_major=None,
            training=True,
        )
        self.assertEqual(cfg["torch_dtype_name"], "float32")
        self.assertFalse(cfg["bf16"])
        self.assertFalse(cfg["fp16"])
        self.assertEqual(cfg["attn_implementation"], "eager")


if __name__ == "__main__":
    unittest.main()
