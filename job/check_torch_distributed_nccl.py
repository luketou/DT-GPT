import ctypes.util
import os
import socket

import torch


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    nccl_env = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("NCCL") or key.startswith("TORCH_NCCL")
    }

    if rank == 0:
        print(f"Hostname: {socket.gethostname()}", flush=True)
        print(f"Torch: {torch.__version__}", flush=True)
        print(f"Torch CUDA: {torch.version.cuda}", flush=True)
        print(f"Torch NCCL: {torch.cuda.nccl.version()}", flush=True)
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        print(f"CUDA device count: {torch.cuda.device_count()}", flush=True)
        print(f"CUDA_VISIBLE_DEVICES: {visible_devices}", flush=True)
        print(f"ctypes nccl lookup: {ctypes.util.find_library('nccl')}", flush=True)
        print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}", flush=True)
        print(f"NCCL env: {nccl_env}", flush=True)

    device_name = torch.cuda.get_device_name(local_rank)
    print(
        f"Rank {rank}/{world_size} local_rank={local_rank} device={device_name}",
        flush=True,
    )
    torch.distributed.barrier()
    if rank == 0:
        print("NCCL barrier probe passed.", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
