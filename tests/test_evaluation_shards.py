import unittest

from pipeline.evaluation_shards import shard_suffix, slice_by_shard


class EvaluationShardTests(unittest.TestCase):
    def test_slices_contiguous_shards(self):
        items = list(range(10))
        metadata = [{"idx": idx} for idx in items]

        shard_items, shard_metadata = slice_by_shard(items, metadata, shard_index=1, num_shards=3)

        self.assertEqual(shard_items, [3, 4, 5])
        self.assertEqual(shard_metadata, [{"idx": 3}, {"idx": 4}, {"idx": 5}])

    def test_last_shard_gets_remainder(self):
        items = list(range(10))
        shard_items, _ = slice_by_shard(items, None, shard_index=2, num_shards=3)

        self.assertEqual(shard_items, [6, 7, 8, 9])

    def test_rejects_invalid_shard(self):
        with self.assertRaises(ValueError):
            slice_by_shard([1, 2], None, shard_index=2, num_shards=2)

    def test_rejects_metadata_length_mismatch(self):
        with self.assertRaises(ValueError):
            slice_by_shard([1, 2], [{"idx": 1}], shard_index=0, num_shards=1)

    def test_shard_suffix_is_stable(self):
        self.assertEqual(shard_suffix(2, 8), "_shard_002_of_008")
        self.assertEqual(shard_suffix(0, 1), "")


if __name__ == "__main__":
    unittest.main()
