import logging
import os
import resource
import time

from joblib import Parallel, delayed
from tqdm import tqdm


logger = logging.getLogger(__name__)


def resolve_df_conversion_n_jobs(n_jobs=None):
    try:
        if n_jobs is not None:
            resolved_n_jobs = int(n_jobs)
        else:
            resolved_n_jobs = int(os.environ.get("DTGPT_DF_CONVERSION_N_JOBS", "1"))
    except ValueError as error:
        raise ValueError("DTGPT_DF_CONVERSION_N_JOBS must be a positive integer.") from error

    if resolved_n_jobs < 1:
        raise ValueError("DTGPT_DF_CONVERSION_N_JOBS must be a positive integer.")

    return resolved_n_jobs


def _rss_megabytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 10_000_000:
        return rss / (1024 * 1024)
    return rss / 1024


def log_memory_usage(label):
    logger.info("Memory usage at %s: maxrss=%.1f MB", label, _rss_megabytes())


def iter_converted_tuples(list_of_data_tuples, conversion_function, log_every=100):
    total = len(list_of_data_tuples)
    started_at = time.monotonic()
    log_memory_usage("df-conversion-start")

    for idx, data in enumerate(list_of_data_tuples, start=1):
        if idx == 1 or idx % log_every == 0 or idx == total:
            elapsed = time.monotonic() - started_at
            logger.info(
                "Converting DFs to Strings: %s / %s elapsed=%.1fs",
                idx,
                total,
                elapsed,
            )
            log_memory_usage(f"df-conversion-{idx}-of-{total}")

        string_input, string_output, meta_data = conversion_function(*data)
        yield {
            "input_text": string_input,
            "target_text": string_output,
            "meta_data": meta_data,
        }

    elapsed = time.monotonic() - started_at
    logger.info("Finished converting %s DFs to strings in %.1fs", total, elapsed)
    log_memory_usage("df-conversion-finished")


def process_all_tuples(list_of_data_tuples, conversion_function):
    list_input_strings = []
    list_target_strings = []
    list_meta_data = []

    for record in iter_converted_tuples(list_of_data_tuples, conversion_function, log_every=10):
        list_input_strings.append(record["input_text"])
        list_target_strings.append(record["target_text"])
        list_meta_data.append(record["meta_data"])

    return list_input_strings, list_target_strings, list_meta_data


def process_all_tuples_multiprocessing(list_of_data_tuples, conversion_function, n_jobs=None):
    resolved_n_jobs = resolve_df_conversion_n_jobs(n_jobs)
    logger.info(
        "Converting DFs to Strings with joblib workers: %s for %s tuples",
        resolved_n_jobs,
        len(list_of_data_tuples),
    )

    if resolved_n_jobs == 1:
        return process_all_tuples(list_of_data_tuples, conversion_function)

    results = Parallel(n_jobs=resolved_n_jobs)(
        delayed(conversion_function)(*data) for data in tqdm(list_of_data_tuples)
    )

    list_input_strings, list_target_strings, list_meta_data = zip(*results)
    return list(list_input_strings), list(list_target_strings), list(list_meta_data)
