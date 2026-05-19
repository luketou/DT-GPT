import logging

import pytest

from pipeline.DFConversionHelpers import (
    iter_converted_tuples,
    log_memory_usage,
    process_all_tuples_multiprocessing,
    resolve_df_conversion_n_jobs,
)


def test_resolve_df_conversion_n_jobs_rejects_zero():
    with pytest.raises(ValueError, match="positive integer"):
        resolve_df_conversion_n_jobs(0)


def test_iter_converted_tuples_preserves_order_and_metadata(caplog):
    def convert(value):
        return f"input-{value}", f"target-{value}", {"idx": value}

    caplog.set_level(logging.INFO)

    records = list(iter_converted_tuples([(1,), (2,), (3,)], convert, log_every=2))

    assert records == [
        {"input_text": "input-1", "target_text": "target-1", "meta_data": {"idx": 1}},
        {"input_text": "input-2", "target_text": "target-2", "meta_data": {"idx": 2}},
        {"input_text": "input-3", "target_text": "target-3", "meta_data": {"idx": 3}},
    ]
    assert "Converting DFs to Strings: 2 / 3" in caplog.text


def test_process_all_tuples_multiprocessing_uses_streaming_path_for_one_worker(monkeypatch):
    def fail_if_joblib_parallel_is_called(*args, **kwargs):
        raise AssertionError("joblib Parallel must not be used for n_jobs=1")

    def convert(value):
        return f"input-{value}", f"target-{value}", {"idx": value}

    monkeypatch.setattr("pipeline.DFConversionHelpers.Parallel", fail_if_joblib_parallel_is_called)

    input_strings, target_strings, meta_data = process_all_tuples_multiprocessing(
        [(1,), (2,)],
        convert,
        n_jobs=1,
    )

    assert input_strings == ["input-1", "input-2"]
    assert target_strings == ["target-1", "target-2"]
    assert meta_data == [{"idx": 1}, {"idx": 2}]


def test_log_memory_usage_does_not_fail_without_psutil(caplog):
    caplog.set_level(logging.INFO)
    log_memory_usage("unit-test")
    assert "unit-test" in caplog.text
