# MIMIC LoRA V100 NCCL Barrier Validation

Distributed smoke command:

```bash
sbatch --time=00:10:00 --export=ALL,DTGPT_SWEEP_CONFIGS=32,64,16,0.01,8e-6,DTGPT_NUM_SAMPLES_TO_GENERATE=1,DTGPT_MAX_NEW_TOKENS=16,DTGPT_SEQ_MAX_LEN=256,DTGPT_RUN_DISTRIBUTED_SMOKE_CHECK=1 job/submit_mimic_lora_sweep_v100.sh
```

Result:

```text
Submitted batch job 35752
Running NCCL distributed smoke check
Torch: 2.1.1+cu121
Torch CUDA: 12.1
Torch NCCL: (2, 18, 1)
ctypes nccl lookup: None
LD_LIBRARY_PATH: /home/r15543056/miniconda3/envs/dtgpt/lib/python3.8/site-packages/nvidia/nccl/lib:/opt/cuda-12.1.0_530.30.02/lib64:...
NCCL env: {'NCCL_ASYNC_ERROR_HANDLING': '1', 'NCCL_DEBUG': 'INFO'}
ncclInvalidArgument: Invalid value for an argument.
Last error:
Invalid config blocking attribute value -2147483648
torch.distributed.DistBackendError: NCCL error in: ../torch/csrc/distributed/c10d/NCCLUtils.hpp:219, invalid argument, NCCL version 2.14.3
ChildFailedError
```

Single-process fallback command:

```bash
env DTGPT_SWEEP_CONFIGS=32,64,1,0.01,8e-6 sbatch --time=00:20:00 --gres=gpu:1 --export=ALL,DTGPT_USE_DEEPSPEED=0,DTGPT_NPROC_PER_NODE=1,DTGPT_NUM_SAMPLES_TO_GENERATE=1,DTGPT_MAX_NEW_TOKENS=16,DTGPT_SEQ_MAX_LEN=256 job/submit_mimic_lora_sweep_v100.sh
```

Result:

```text
Submitted batch job 35760
MIMIC local setup smoke check passed.
Training mode: LoRA
Start training
```
