import os
import unittest
from unittest.mock import patch

from pipeline import DFConversionHelpers as helpers


class TestDFConversionWorkerResolution(unittest.TestCase):
    def test_defaults_to_single_worker_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(helpers.resolve_df_conversion_n_jobs(), 1)

    def test_uses_positive_integer_from_environment(self):
        with patch.dict(os.environ, {"DTGPT_DF_CONVERSION_N_JOBS": "2"}, clear=True):
            self.assertEqual(helpers.resolve_df_conversion_n_jobs(), 2)

    def test_rejects_zero_from_environment(self):
        with patch.dict(os.environ, {"DTGPT_DF_CONVERSION_N_JOBS": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "DTGPT_DF_CONVERSION_N_JOBS"):
                helpers.resolve_df_conversion_n_jobs()

    def test_rejects_non_integer_from_environment(self):
        with patch.dict(os.environ, {"DTGPT_DF_CONVERSION_N_JOBS": "abc"}, clear=True):
            with self.assertRaisesRegex(ValueError, "DTGPT_DF_CONVERSION_N_JOBS"):
                helpers.resolve_df_conversion_n_jobs()

    def test_explicit_argument_overrides_environment(self):
        with patch.dict(os.environ, {"DTGPT_DF_CONVERSION_N_JOBS": "4"}, clear=True):
            self.assertEqual(helpers.resolve_df_conversion_n_jobs(2), 2)


class _FakeParallel:
    last_n_jobs = None

    def __init__(self, n_jobs):
        _FakeParallel.last_n_jobs = n_jobs

    def __call__(self, delayed_calls):
        return [call() for call in delayed_calls]


class _FakeDelayedCall:
    def __init__(self, func, args):
        self.func = func
        self.args = args

    def __call__(self):
        return self.func(*self.args)


def _fake_delayed(func):
    def wrapper(*args):
        return _FakeDelayedCall(func, args)
    return wrapper


def _convert(value):
    return f"input-{value}", f"target-{value}", {"value": value}


class TestProcessAllTuplesMultiprocessing(unittest.TestCase):
    def test_passes_resolved_worker_count_to_joblib(self):
        _FakeParallel.last_n_jobs = None
        with patch.dict(os.environ, {"DTGPT_DF_CONVERSION_N_JOBS": "2"}, clear=True):
            with patch.object(helpers, "Parallel", _FakeParallel), patch.object(helpers, "delayed", _fake_delayed):
                inputs, targets, metas = helpers.process_all_tuples_multiprocessing([(1,), (2,)], _convert)

        self.assertEqual(_FakeParallel.last_n_jobs, 2)
        self.assertEqual(inputs, ("input-1", "input-2"))
        self.assertEqual(targets, ("target-1", "target-2"))
        self.assertEqual(metas, ({"value": 1}, {"value": 2}))


if __name__ == "__main__":
    unittest.main()
