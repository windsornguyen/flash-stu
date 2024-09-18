import logging
import os
import random
import socket
import sys
from packaging.version import parse as version_parse
from functools import partial
from safetensors.torch import save_file, load_file
import numpy as np
import psutil
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
import torch.nn as nn
from glob import glob
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import _module_wrap_policy, size_based_auto_wrap_policy

from flash_stu import STU
from flash_stu.modules.attention import Attention
from flash_stu.modules.swiglu import MLP


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seeds(seed: int, cuda_deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seeds set to {seed}")


def setup_distributed(seed: int = 1337) -> tuple[torch.device, int, int, int, bool]:
    if not dist.is_available():
        raise RuntimeError("Distributed package not available!")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distributed training!")

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    main_process = rank == 0

    # Set up devices
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    set_seeds(seed + rank)

    if main_process:
        logger.info(f"Main process initialized on {socket.gethostname()}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(
            f"World info: size={world_size}, rank={rank}, local_rank={local_rank}"
        )
        log_system_info(world_size, rank)

    return device, local_rank, rank, world_size, main_process


def log_system_info(world_size: int, rank: int):
    logger.info(f"System info for rank {rank}:")
    logger.info(f"CPU count: {psutil.cpu_count()}")
    logger.info(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    logger.info(
        f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB"
    )

    # Log local GPU count and world size for clarity
    local_gpu_count = torch.cuda.device_count()
    logger.info(f"Local GPU count (rank {rank}): {local_gpu_count}")
    logger.info(f"Total GPU count across all nodes: {world_size * local_gpu_count}")

    # Log specific GPU properties for this node (local GPUs)
    for i in range(local_gpu_count):
        logger.info(f"GPU {i} (rank {rank}) name: {torch.cuda.get_device_name(i)}")
        logger.info(
            f"GPU {i} (rank {rank}) memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB"
        )


def find_checkpoint(log_dir: str) -> str:
    model_pattern = os.path.join(log_dir, "model_*.safetensors")
    misc_pattern = os.path.join(log_dir, "other_checkpoints_*.pt")
    model_checkpoints = glob(model_pattern)
    misc_checkpoints = glob(misc_pattern)
    if not model_checkpoints or not misc_checkpoints:
        return None
    latest_checkpoint = max(
        model_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    misc_checkpoint = max(
        misc_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    return latest_checkpoint, misc_checkpoint


def load_checkpoint(model_path: str, misc_path: str, model, optimizer, device):
    model_checkpoint = load_file(model_path)
    model.load_state_dict(model_checkpoint)
    model.to(device)

    misc_checkpoint = torch.load(misc_path, map_location=device, weights_only=True)
    model.config = misc_checkpoint["config"]
    optimizer.load_state_dict(misc_checkpoint["optimizer"])

    step = misc_checkpoint["step"]
    val_loss = misc_checkpoint["val_loss"]

    return model, optimizer, step, val_loss


def save_checkpoint(
    model_checkpoint, optim_checkpoint, config, step, best_val_loss, log_dir
):
    model_checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.safetensors")
    other_checkpoint_path = os.path.join(log_dir, f"other_checkpoints_{step:05d}.pt")

    save_file(model_checkpoint, model_checkpoint_path)

    other_checkpoint = {
        "config": config,
        "optimizer": optim_checkpoint,
        "step": step,
        "val_loss": best_val_loss,
    }
    torch.save(other_checkpoint, other_checkpoint_path)

    logging.info(
        f"Validation loss improved at step {step}! Save the model to {model_checkpoint_path}, misc data to {other_checkpoint_path}."
    )


def setup_fsdp(
    model: nn.Module,
    mixed_precision: bool = True,
    use_cpu_offload: bool = False,
    sharding_strategy: str = "full_shard",
    auto_wrap_policy: str = "partial",
    backward_prefetch: str = "backward_pre",
    forward_prefetch: bool = False,
    sync_module_states: bool = True,
    use_orig_params: bool = True,
    device_id: int = None,
    precision: dict = None,
    fsdp_modules: list = None,
    use_activation_checkpointing: bool = True,
) -> tuple[FSDP, dict]:
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise RuntimeError("CUDA and NCCL must be available for FSDP setup")

    fsdp_params = {}

    bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and version_parse(torch.version.cuda) >= version_parse("11.0")
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    # Set up mixed precision
    if mixed_precision and precision:
        param_dtype = precision.get(
            "param", torch.bfloat16 if bf16_ready else torch.float32
        )
        reduce_dtype = precision.get(
            "reduce", torch.bfloat16 if bf16_ready else torch.float32
        )
        buffer_dtype = precision.get(
            "buffer", torch.bfloat16 if bf16_ready else torch.float32
        )

        if isinstance(param_dtype, str):
            param_dtype = getattr(torch, param_dtype)
        if isinstance(reduce_dtype, str):
            reduce_dtype = getattr(torch, reduce_dtype)
        if isinstance(buffer_dtype, str):
            buffer_dtype = getattr(torch, buffer_dtype)

        fsdp_params["mixed_precision"] = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

    # Set up CPU offloading
    if use_cpu_offload:
        fsdp_params["use_cpu_offload"] = CPUOffload(offload_params=True)

    # Set up sharding strategy
    if sharding_strategy == "full_shard":
        fsdp_params["sharding_strategy"] = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "shard_grad_op":
        fsdp_params["sharding_strategy"] = ShardingStrategy.SHARD_GRAD_OP
    elif sharding_strategy == "no_shard":
        fsdp_params["sharding_strategy"] = ShardingStrategy.NO_SHARD
    else:
        raise ValueError(f"Invalid sharding strategy: {sharding_strategy}")

    # Set up backward prefetch
    if backward_prefetch == "backward_pre":
        fsdp_params["backward_prefetch"] = BackwardPrefetch.BACKWARD_PRE
    elif backward_prefetch == "backward_post":
        fsdp_params["backward_prefetch"] = BackwardPrefetch.BACKWARD_POST
    elif backward_prefetch is not None:
        raise ValueError(f"Invalid backward prefetch option: {backward_prefetch}")

    # Set up other parameters
    fsdp_params["forward_prefetch"] = forward_prefetch
    fsdp_params["sync_module_states"] = sync_module_states
    fsdp_params["use_orig_params"] = use_orig_params

    if device_id is None:
        device_id = torch.cuda.current_device()
    fsdp_params["device_id"] = device_id

    # Set up auto wrap policy
    fsdp_modules_set = set(
        eval(module) if isinstance(module, str) else module for module in fsdp_modules
    )
    if auto_wrap_policy == "partial":
        fsdp_params["auto_wrap_policy"] = partial(
            _module_wrap_policy, module_classes=fsdp_modules_set
        )
    elif auto_wrap_policy == "size_based":
        fsdp_params["auto_wrap_policy"] = size_based_auto_wrap_policy
    else:
        raise ValueError(f"Invalid auto wrap policy: {auto_wrap_policy}")

    # Apply activation checkpointing
    if use_activation_checkpointing:
        check_fn = lambda submodule: isinstance(submodule, tuple(fsdp_modules_set))
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=check_fn,
        )

    # Wrap the model with FSDP
    fsdp_model = FSDP(model, **fsdp_params)

    return fsdp_model


def cleanup_distributed(rank: int):
    if dist.is_initialized():
        logging.info(f"[Rank {rank}]: Finished training.")
        logging.info(f"[Rank {rank}]: Waiting for other processes to finish...")
        dist.barrier()
        dist.destroy_process_group()
