def shard_suffix(shard_index, num_shards):
    if num_shards == 1:
        return ""
    return "_shard_{:03d}_of_{:03d}".format(shard_index, num_shards)


def slice_by_shard(items, metadata, shard_index, num_shards):
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")

    if metadata is not None and len(metadata) != len(items):
        raise ValueError("metadata length must match items length")

    start = (len(items) * shard_index) // num_shards
    end = (len(items) * (shard_index + 1)) // num_shards

    shard_items = items[start:end]
    shard_metadata = None if metadata is None else metadata[start:end]

    return shard_items, shard_metadata
