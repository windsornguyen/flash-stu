import json
import logging
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from dataloader import DistributedDataloader
from distributed import (
    cleanup_distributed,
    find_checkpoint,
    load_checkpoint,
    save_checkpoint,
    setup_distributed,
    setup_fsdp
)
from flash_stu.config import FlashSTUConfig
from flash_stu.model import FlashSTU
from flash_stu.utils.stu_utils import get_spectral_filters
from flash_stu.utils.training import GarbageCollection, linear_decay_with_warmup

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss as CrossEntropyLoss
except ImportError as e:
    print(f"Unable to import Triton-based cross entropy loss: {e}. Falling back to PyTorch implementation.")
    from torch.nn import CrossEntropyLoss


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@record
def main():
    torch.set_float32_matmul_precision("high")
    device, local_rank, rank, world_size, main_process = setup_distributed(seed=1337)
    gc_handler = GarbageCollection()

    with open("config.json", "r") as file:
        config  = json.load(file)

    # Model configurations
    n_embd             = config['n_embd']
    n_heads            = config['n_heads']
    n_layers           = config['n_layers']
    seq_len            = config['seq_len']
    window_size        = config['window_size']
    vocab_size         = config['vocab_size']
    mlp_scale          = config['mlp_scale']
    bias               = config['bias']
    dropout            = config['dropout']
    num_eigh           = config['num_eigh']
    use_hankel_L       = config['use_hankel_L']
    
    # Optimizations
    use_flash_fft      = config['use_flash_fft']
    use_approx         = config['use_approx']
    use_attn           = config['use_attn']
    softcap            = config['softcap']
    torch_compile      = config['torch_compile']

    # Training configurations
    dilation           = config['dilation']
    warmup_steps       = config['warmup_steps'] // dilation
    eval_period        = config['eval_period'] // dilation
    save_period        = config['save_period'] // dilation
    num_epochs         = config['num_epochs']    
    max_lr             = config['max_lr']
    min_lr             = config['min_lr']
    max_norm           = config['max_norm']

    global_bsz         = config['global_bsz']
    bsz                = config['bsz']
    assert (
        global_bsz % (bsz * seq_len * world_size) == 0
    ), f"global_bsz ({global_bsz}) must be divisible by bsz * seq_len * world_size ({bsz * seq_len * world_size}),"
    f" got {global_bsz % (bsz * seq_len * world_size)}"
    gradient_accumulation_steps = global_bsz // (bsz * seq_len * world_size)

    # Distributed
    fsdp               = config['fsdp']
    ddp                = config['ddp']
    assert not (fsdp and ddp), "FSDP and DDP are both enabled which is not allowed"

    distributed        = (fsdp or ddp) and world_size > 1
    cache_enabled      = not ddp
    
    mixed_precision    = config['mixed_precision']
    torch_dtype        = getattr(torch, config['torch_dtype'])
    use_cpu_offload    = config['use_cpu_offload']
    sharding_strategy  = config['sharding_strategy']
    auto_wrap_policy   = config['auto_wrap_policy']
    backward_prefetch  = config['backward_prefetch']
    forward_prefetch   = config['forward_prefetch']
    sync_module_states = config['sync_module_states']
    use_orig_params    = config['use_orig_params']
    device_id          = config['device_id']
    precision          = config['precision']
    fsdp_modules       = config['fsdp_modules']
    use_activation_checkpointing = config['use_activation_checkpointing']

    if main_process:
        logging.info(f"Training config: {config}\n")

    if world_size == 1 and fsdp:
        if main_process:
            logging.info("World size is 1, disabling sharding.")
        sharding_strategy = "no_shard"

    config = FlashSTUConfig(
        bsz=bsz,
        n_embd=n_embd,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
        window_size=window_size,
        vocab_size=vocab_size,
        mlp_scale=mlp_scale,
        bias=bias,
        dropout=dropout,
        num_eigh=num_eigh,
        use_hankel_L=use_hankel_L,
        use_flash_fft=use_flash_fft,
        use_approx=use_approx,
        use_attn=use_attn,
        softcap=softcap,
        torch_dtype=torch_dtype,
    )
    fsdp_params = {
        "mixed_precision": mixed_precision,
        "use_cpu_offload": use_cpu_offload,
        "sharding_strategy": sharding_strategy,
        "auto_wrap_policy": auto_wrap_policy,
        "backward_prefetch": backward_prefetch,
        "forward_prefetch": forward_prefetch,
        "sync_module_states": sync_module_states,
        "use_orig_params": use_orig_params,
        "device_id": device_id,
        "precision": precision,
        "fsdp_modules": fsdp_modules,
        "use_activation_checkpointing": use_activation_checkpointing,
    }
    
    phi = get_spectral_filters(seq_len, num_eigh, use_hankel_L, device, torch_dtype)
    model = FlashSTU(config, phi)
    if torch_compile:
        model = torch.compile(model)
        if main_process:
            logging.info(f"PyTorch Compiler Enabled?: {torch_compile}")
    model = model.to(device)

    if fsdp:
        model = setup_fsdp(model, **fsdp_params)
    if ddp:
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    state_dict_options = StateDictOptions(
        full_state_dict=True, 
        cpu_offload=True,
    )
    torch.serialization.add_safe_globals([FlashSTUConfig])

    use_scaler = torch_dtype == torch.float16
    if use_scaler:
        scaler = ShardedGradScaler() if fsdp else GradScaler()
    optimizer = AdamW(
        model.parameters(),
        lr=max_lr,
        fused=torch.cuda.is_available(),
    )
    loss_fn = CrossEntropyLoss()

    # Create the log directory to write checkpoints to and log to
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)

    checkpoints = find_checkpoint(log_dir)
    if checkpoints:
        model_checkpoint, misc_checkpoint = checkpoints
        model, optimizer, start_step, best_val_loss = load_checkpoint(
            model_checkpoint, misc_checkpoint, model, optimizer, device
        )
        log_mode = "a"  # Append to the log if resuming
        if main_process:
            logging.info(f"Resuming from checkpoint: {model_checkpoint}")
            logging.info(f"Starting from step: {start_step}")
            logging.info(f"Best validation loss: {best_val_loss}")
    else:
        start_step = 0
        best_val_loss = float('inf')
        log_mode = "w"  # Create a new log if starting fresh

    if main_process:
        log_file = os.path.join(log_dir, "log.txt")
        with open(log_file, log_mode) as f:
            pass

    # Data loader section
    # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size ~0.5M tokens
    # This is dataset and batch size dependent.
    dataset = "data/fineweb-edu-10B"
    total_tokens = 10_000_000_000
    num_steps = total_tokens // global_bsz # Number of steps for one epoch
    max_steps = num_steps * num_epochs

    if main_process:
        logging.info(f"Total (desired) batch size: {global_bsz}")
        logging.info(
            f"=> Calculated gradient accumulation steps: {gradient_accumulation_steps}"
        )
        logging.info(f"Training on {max_steps} steps")

    train_loader = DistributedDataloader(
        bsz=bsz,
        seq_len=seq_len, 
        rank=rank, 
        world_size=world_size, 
        dataset=dataset, 
        split="train", 
        main_process=main_process,
    )
    val_loader = DistributedDataloader(
        bsz=bsz,
        seq_len=seq_len, 
        rank=rank, 
        world_size=world_size, 
        dataset=dataset, 
        split="val", 
        main_process=main_process,
    )

    for step in range(start_step + 1, max_steps + 1):
        gc_handler.run(step)
        torch.cuda.empty_cache()
        epoch = step // num_steps
        last_step = step % num_steps == 0

        if step == 1 or step % num_steps == 1:
            if main_process:
                logging.info(f"Starting epoch {epoch}")
            train_loader.set_epoch(epoch)

        t0 = time.perf_counter()

        if step == 1 or step % (eval_period // dilation) == 0 or last_step:
            val_loss = 0.0
            val_steps = 20 # Arbitrarily set to reduce long evaluations
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                for i, batch in zip(range(val_steps), val_loader, strict=False):
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    if torch_dtype != torch.float32:
                        with autocast(device_type=device.type, dtype=torch_dtype, cache_enabled=cache_enabled):
                            preds = model(inputs)
                    else:
                        preds = model(inputs)

                    loss = loss_fn(preds.flatten(0, 1), targets.flatten(0, 1))
                    loss = loss / val_steps
                    val_loss += loss.detach().float()

            if distributed:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

            if main_process:
                logging.info(f"Validation loss: {val_loss.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss.item():.4f}\n")

            if step > 0 and (step % save_period == 0 or last_step):
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    dist.barrier()
                    model_checkpoint, optim_checkpoint = get_state_dict(
                        model, optimizer, options=state_dict_options
                    )
                    if main_process:
                        save_checkpoint(model_checkpoint, optim_checkpoint, config, step, best_val_loss, log_dir)

        model.train()
        train_loss = 0.0
        for micro_step, batch in zip(range(gradient_accumulation_steps), train_loader, strict=False):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            last_micro_step = micro_step == gradient_accumulation_steps - 1

            if (fsdp or ddp) and not last_micro_step:
                context = model.no_sync()
            else:
                context = nullcontext()
                
            with context:
                if torch_dtype != torch.float32:
                    with autocast(device_type=device.type, dtype=torch_dtype, cache_enabled=cache_enabled):
                        preds = model(inputs)
                else:
                    preds = model(inputs)

                loss = loss_fn(preds.flatten(0, 1), targets.flatten(0, 1))
                loss = loss / gradient_accumulation_steps
                train_loss += loss.detach().float()
                scaler.scale(loss).backward() if use_scaler else loss.backward()

        if distributed:
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)

        if use_scaler:
            scaler.unscale_(optimizer)

        if fsdp:
            norm = model.clip_grad_norm_(max_norm)
        else:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        lr = linear_decay_with_warmup(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dt = t1 - t0
        toks_processed = (
            train_loader.bsz
            * train_loader.seq_len
            * gradient_accumulation_steps
            * world_size
        )
        toks_per_sec = toks_processed / dt          

        if main_process:
            log_message = (
                f"step {step:5d} | "
                f"loss: {train_loss:.6f} | "
                f"lr {lr:.4e} | "
                f"norm: {norm:.4f} | "
                f"dt: {dt*1000:.2f}ms | "
                f"tok/s: {toks_per_sec:.2f}"
            )
            print(log_message)
            
            with open(log_file, "a") as f:
                f.write(f"{step} train {train_loss:.6f} lr {lr:.4e} norm {norm:.4f} dt {dt*1000:.2f} tok/s {toks_per_sec:.2f}\n")

    cleanup_distributed(rank)

if __name__ == "__main__":
    main()
