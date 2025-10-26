import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer

from model_mp import GPTConfig, GPT_mp
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2-model-parallel' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

model_parallel = 1  # number of model shards to split into

backend = 'gloo'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# DDP settings
ddp = False
# These are set by torchrun
ddp_rank = int(os.environ.get('RANK', 0))
ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0))
ddp_world_size = int(os.environ.get('WORLD_SIZE', 1))
master_process = ddp_rank == 0
seed_offset = ddp_rank # each process gets a different seed

# Master process, the data loader, must run on CPU
device = 'cpu'
device_type = 'cpu'
from train import get_batch, get_lr, data_dir  # load data onto the CPU now


# Re-define estimate_loss because the one in train.py assumes a local ctx
@torch.no_grad()
def estimate_loss(model, ctx):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



def master():
    print("Master rank 0 starting...")
    torch.manual_seed(1337 + seed_offset)

    # Master runs on CPU, so ctx is null
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    # Autocast context is null on CPU (master)
    # The remote shards will create their own 'cuda' autocast contexts
    ctx = nullcontext()

    os.makedirs(out_dir, exist_ok=True)

    iter_num = 0
    best_val_loss = 1e9

    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, vocab_size=None, dropout=dropout)  # start with model_args from command line
    if init_from != 'scratch':
        raise NotImplementedError("Only scratch init is implemented in model parallel")

    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)


    model_parallel_size = config['model_parallel']
    mp_world_size = 1

    if model_parallel_size > 1:
        if model_parallel_size != ddp_world_size:
            print(f"ERROR: Mismatch in parallel size.")
            print(f"  torchrun launch detected {ddp_world_size}")
            print(f"  --model_parallel argument requested {model_parallel_size}")
            print(
                f"  torchrun --nproc_per_node={model_parallel_size} ... train_mp.py --model_parallel={model_parallel_size} ...")
            rpc.shutdown()
            return
        else:
            mp_world_size = model_parallel_size

    elif ddp_world_size > 1:
        print(f"Using WORLD_SIZE={ddp_world_size} from torchrun")
        mp_world_size = ddp_world_size
    else:
        print("Training in single gpu")
        mp_world_size = 1

    if mp_world_size <= n_layer:
        print(f"Initializing model parallel across {mp_world_size} processes")
    else:
        print(f"ERROR: Model parallel size ({mp_world_size}) is larger than n_layer ({n_layer}).")
        rpc.shutdown()
        return

    model = GPT_mp(gptconf, model_parallel=mp_world_size)

    param_rrefs = model.parameter_rrefs()
    optimizer = DistributedOptimizer(
        torch.optim.AdamW,
        param_rrefs,
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay
    )

    # logging
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)


    # training loop
    X, Y = get_batch('train')  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    running_mfu = -1.0


    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss(model, ctx)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                })

        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        context_id = None
        for micro_step in range(gradient_accumulation_steps):
            with dist_autograd.context() as ctx_id:
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

                # Backward pass
                dist_autograd.backward(ctx_id, [loss])

                if micro_step == 0:
                    # Store the context_id from the first micro_batch
                    context_id = ctx_id
                else:
                    # Merge subsequent micro_batch grads into the first context
                    dist_autograd.merge_contexts(context_id, ctx_id)

            # prefetch next batch
            if micro_step < gradient_accumulation_steps - 1:
                X, Y = get_batch('train')

        # clip the gradient
        if grad_clip != 0.0:
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            pass  # too lazy to implement and debug this in rpc

        # step the optimizer
        optimizer.step(context_id)
        optimizer.zero_grad(set_to_none=True)

        X, Y = get_batch('train')

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break

def worker():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    print(f"Starting process {rank}/{world_size} (local rank {local_rank})")

    master_addr = os.environ['MASTER_ADDR']
    master_port = int(os.environ['MASTER_PORT']) + 1  # avoid port conflicts

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)

    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=120
    )

    if rank == 0:
        # Master process 0
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        master()
    else:
        # Worker process
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    # Block until all processes are done
    print(f"Rank {rank} shutting down RPC.")
    rpc.shutdown()
    print(f"Rank {rank} finished.")

if __name__ == "__main__":
    worker()