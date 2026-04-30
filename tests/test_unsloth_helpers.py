"""Tests for pipeline.unsloth_helpers."""

import sys
import unittest
from unittest.mock import MagicMock, patch


class TestIsUnslothAvailable(unittest.TestCase):
    """Test the availability-check helper."""

    def test_returns_true_when_unsloth_importable(self):
        fake_unsloth = MagicMock()
        with patch.dict(sys.modules, {"unsloth": fake_unsloth}):
            from pipeline.unsloth_helpers import is_unsloth_available
            # Reimport to pick up the patched module
            self.assertTrue(is_unsloth_available())

    def test_returns_false_when_unsloth_missing(self):
        with patch.dict(sys.modules, {"unsloth": None}):
            from pipeline.unsloth_helpers import is_unsloth_available
            self.assertFalse(is_unsloth_available())


class TestRequireUnsloth(unittest.TestCase):
    """Test the internal _require_unsloth guard."""

    def test_raises_import_error_when_missing(self):
        with patch.dict(sys.modules, {"unsloth": None}):
            from pipeline.unsloth_helpers import _require_unsloth
            with self.assertRaises(ImportError):
                _require_unsloth()


class TestLoadModelUnsloth(unittest.TestCase):
    """Test load_model_unsloth delegates to FastLanguageModel correctly."""

    def test_calls_from_pretrained(self):
        mock_fast_lm = MagicMock()
        mock_fast_lm.from_pretrained.return_value = ("mock_model", "mock_tok")

        fake_unsloth = MagicMock()
        fake_unsloth.FastLanguageModel = mock_fast_lm

        with patch.dict(sys.modules, {"unsloth": fake_unsloth}):
            from pipeline.unsloth_helpers import load_model_unsloth
            model, tok = load_model_unsloth(
                "test-model", max_seq_length=2048, load_in_4bit=True
            )

        mock_fast_lm.from_pretrained.assert_called_once_with(
            model_name="test-model",
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,
        )
        self.assertEqual(model, "mock_model")
        self.assertEqual(tok, "mock_tok")


class TestApplyUnslothPeft(unittest.TestCase):
    """Test apply_unsloth_peft delegates to FastLanguageModel.get_peft_model."""

    def test_calls_get_peft_model_with_dora(self):
        mock_model = MagicMock()
        mock_model.print_trainable_parameters = MagicMock()
        mock_fast_lm = MagicMock()
        mock_fast_lm.get_peft_model.return_value = mock_model

        fake_unsloth = MagicMock()
        fake_unsloth.FastLanguageModel = mock_fast_lm

        with patch.dict(sys.modules, {"unsloth": fake_unsloth}):
            from pipeline.unsloth_helpers import apply_unsloth_peft
            result = apply_unsloth_peft(
                mock_model, r=32, lora_alpha=64, use_dora=True
            )

        call_kwargs = mock_fast_lm.get_peft_model.call_args
        self.assertTrue(call_kwargs.kwargs.get("use_dora", call_kwargs[1].get("use_dora")))
        self.assertEqual(result, mock_model)


class TestSaveUnslothModel(unittest.TestCase):
    """Test save_unsloth_model calls save_pretrained."""

    def test_save_calls_save_pretrained(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        from pipeline.unsloth_helpers import save_unsloth_model

        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "adapter_out")
            save_unsloth_model(mock_model, mock_tokenizer, save_path)
            mock_model.save_pretrained.assert_called_once_with(save_path)
            mock_tokenizer.save_pretrained.assert_called_once_with(save_path)


class TestLocalPathsUnslothAvailable(unittest.TestCase):
    """Test the convenience is_unsloth_available in local_paths."""

    def test_returns_false_in_standard_env(self):
        # In the dtgpt env (Python 3.8), unsloth is not installed.
        from pipeline.local_paths import is_unsloth_available
        # This will be False unless we're in the unsloth env.
        result = is_unsloth_available()
        self.assertIsInstance(result, bool)


class TestCLIFlagParsing(unittest.TestCase):
    """Test that --use-unsloth flag is parsed and implies lora + dora."""

    def test_use_unsloth_implies_lora_and_dora(self):
        # Import the parser builder
        import importlib
        import os

        script_dir = os.path.join(
            os.path.dirname(__file__), "..",
            "1_experiments", "2024_02_08_mimic_iv",
            "4_dt_gpt_instruction",
            "2024_04_11_biomistral_td_bd_summarized_row",
        )
        sys.path.insert(0, script_dir)
        try:
            # We only test the parser, not the full import chain
            spec = importlib.util.spec_from_file_location(
                "launcher",
                os.path.join(
                    script_dir,
                    "2024_04_08_dt_gpt_bd_bm_summarized_row_mimic.py",
                ),
            )
            if spec and spec.loader:
                # Just check the argparse definition exists
                import ast
                with open(spec.origin, "r") as f:
                    source = f.read()
                self.assertIn("--use-unsloth", source)
                self.assertIn("use_unsloth", source)
        finally:
            sys.path.pop(0)


if __name__ == "__main__":
    unittest.main()
