"""
Distributed training utilities.

This module centralizes DistributedDataParallel setup to avoid duplication across
training scripts.
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from omegaconf import DictConfig


def set_up_ddp(config: DictConfig) -> tuple[int, int]:
    """
    Set up DistributedDataParallel for multi-gpu training using torchrun.

    Args:
        config (DictConfig): Configuration object containing settings for the model.

    Returns:
        tuple[int, int]: local_rank (rank of the GPU within the multi-process
                        execution), world_size (total number of GPUs used for training)
    """
    # Check if CUDA is available before any GPU operations
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check your CUDA installation."
        )

    # Check available GPUs in the system
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No CUDA GPUs are available.")

    # Get distributed training environment variables set by torchrun / Slurm.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # GPU selection is owned entirely by the launcher (Slurm ``srun`` /
    # torchrun), which exports CUDA_VISIBLE_DEVICES for the allocation. We never
    # override it from config: doing so breaks Slurm GPU isolation and, on
    # MIG-enabled nodes, remaps integer indices onto MIG instances (causing
    # 19.5 GB OOMs). DDP is sized by the visible device count and each rank
    # binds to its ``local_rank`` within that visible set.
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    # Bind this rank to a visible device. ``local_rank % available_gpus`` works
    # for both launch styles the cluster uses:
    #   - torchrun (all allocated GPUs visible to every worker): a no-op, since
    #     local_rank < available_gpus — including multi-node, where local_rank
    #     is node-local.
    #   - Slurm ``srun --gpus-per-task=1`` (each task sees a single GPU): maps
    #     any local_rank onto the one visible device, avoiding an "invalid
    #     device ordinal" error.
    # We do NOT hard-fail when world_size/local_rank exceeds the visible count:
    # under per-task isolation and multi-node DDP the global world size
    # legitimately exceeds the per-process visible device count.
    local_device = local_rank % available_gpus

    try:
        torch.cuda.set_device(local_device)
    except RuntimeError as e:
        print(f"❌ Failed to set CUDA device {local_device}: {e}")

    # Initialize the process group
    backend = config.get("training", {}).get("distributed", {}).get("backend", "nccl")

    # Use gloo backend if NCCL is not available or for debugging
    if backend == "nccl" and not torch.distributed.is_nccl_available():
        backend = "gloo"

    # Resolve timeout (minutes) from config, default to 30 minutes
    timeout_minutes = (
        config.get("training", {}).get("distributed", {}).get("timeout_minutes", 30)
    )

    # Anchor NCCL to this process's CUDA device. Without this, dist.barrier()
    # infers the device from the current context and can pick the wrong one on
    # single-host multi-GPU setups, producing the "barrier(): using the device
    # under current context" warning and intermittent post-training deadlocks.
    init_kwargs: dict[str, object] = {
        "backend": backend,
        "rank": global_rank,
        "world_size": world_size,
        "timeout": timedelta(minutes=int(timeout_minutes)),
    }
    if backend == "nccl":
        init_kwargs["device_id"] = torch.device(f"cuda:{local_device}")

    try:
        torch.distributed.init_process_group(**init_kwargs)
    except RuntimeError as e:
        print(f"❌ Failed to initialize process group: {e}")
        if backend == "nccl":
            print("🔄 Trying with gloo backend as fallback...")
            torch.distributed.init_process_group(
                backend="gloo",
                rank=global_rank,
                world_size=world_size,
                timeout=timedelta(minutes=int(timeout_minutes)),
            )
            backend = "gloo"
        else:
            raise

    if is_master_process():
        print(f"🔍 System has {available_gpus} visible GPUs")
        print(f"🔧 CUDA_VISIBLE_DEVICES (launcher-allocated): {cuda_visible_devices}")

    return local_rank, world_size


def is_master_process() -> bool:
    """
    Check if current process is the master process (rank 0 or non-distributed).

    Returns:
        bool: True if master or not initialized, else False.
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def cleanup_distributed() -> None:
    """
    Clean up distributed training setup.
    """
    dist.destroy_process_group()
