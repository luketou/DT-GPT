import time
from pathlib import Path


SUCCESS_MARKER = "_SUCCESS"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"


def build_dataset_cache_dir(experiment_folder_path, cache_name):
    return Path(experiment_folder_path) / "dataset_cache" / cache_name


def dataset_cache_paths(cache_dir):
    cache_dir = Path(cache_dir)
    return {
        TRAIN_SPLIT: cache_dir / TRAIN_SPLIT,
        VALIDATION_SPLIT: cache_dir / VALIDATION_SPLIT,
    }


def dataset_cache_complete(cache_dir):
    cache_dir = Path(cache_dir)
    paths = dataset_cache_paths(cache_dir)
    return (
        (cache_dir / SUCCESS_MARKER).exists()
        and (paths[TRAIN_SPLIT] / "state.json").exists()
        and (paths[VALIDATION_SPLIT] / "state.json").exists()
    )


def mark_dataset_cache_complete(cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / SUCCESS_MARKER).write_text("ok\n")


def wait_for_dataset_cache_complete(cache_dir, poll_seconds=30, timeout_seconds=None):
    """Wait until a rank-0-built dataset cache is visible on the filesystem."""
    start = time.monotonic()
    while not dataset_cache_complete(cache_dir):
        if timeout_seconds is not None and time.monotonic() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for tokenized dataset cache: {cache_dir}")
        time.sleep(poll_seconds)
