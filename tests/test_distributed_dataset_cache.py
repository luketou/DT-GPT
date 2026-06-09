import tempfile
import unittest
from pathlib import Path

from pipeline.distributed_dataset_cache import (
    MANIFEST_FILE,
    build_dataset_cache_dir,
    dataset_cache_complete,
    dataset_cache_paths,
    dataset_cache_temp_dir,
    mark_dataset_cache_complete,
    wait_for_dataset_cache_complete,
)


class DistributedDatasetCacheTests(unittest.TestCase):
    def test_cache_is_complete_only_after_success_marker_and_split_state_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            paths = dataset_cache_paths(cache_dir)

            self.assertFalse(dataset_cache_complete(cache_dir))

            paths["train"].mkdir(parents=True)
            paths["validation"].mkdir(parents=True)
            (paths["train"] / "state.json").write_text("{}")
            (paths["validation"] / "state.json").write_text("{}")
            self.assertFalse(dataset_cache_complete(cache_dir))

            mark_dataset_cache_complete(cache_dir)
            self.assertFalse(dataset_cache_complete(cache_dir))

            (cache_dir / MANIFEST_FILE).write_text("{}")
            self.assertTrue(dataset_cache_complete(cache_dir))

    def test_build_dataset_cache_dir_uses_stable_run_local_location(self):
        cache_dir = build_dataset_cache_dir("/tmp/example_run", "mimic_train")
        self.assertEqual(cache_dir, Path("/tmp/example_run") / "dataset_cache" / "mimic_train")

    def test_dataset_cache_temp_dir_uses_sibling_temp_directory(self):
        cache_dir = Path("/tmp/example_run") / "dataset_cache" / "mimic_train"
        self.assertEqual(
            dataset_cache_temp_dir(cache_dir),
            Path("/tmp/example_run") / "dataset_cache" / "mimic_train.tmp",
        )

    def test_wait_for_dataset_cache_complete_returns_when_cache_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            paths = dataset_cache_paths(cache_dir)
            paths["train"].mkdir(parents=True)
            paths["validation"].mkdir(parents=True)
            (paths["train"] / "state.json").write_text("{}")
            (paths["validation"] / "state.json").write_text("{}")
            (cache_dir / MANIFEST_FILE).write_text("{}")
            mark_dataset_cache_complete(cache_dir)

            wait_for_dataset_cache_complete(cache_dir, poll_seconds=0, timeout_seconds=1)

    def test_wait_for_dataset_cache_complete_times_out(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(TimeoutError):
                wait_for_dataset_cache_complete(
                    Path(tmpdir) / "missing-cache",
                    poll_seconds=0,
                    timeout_seconds=0,
                )


if __name__ == "__main__":
    unittest.main()
