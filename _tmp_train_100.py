import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import argparse
import uuid
import glob
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from ssinf3 import (
    BASELINE_MLP_PARAMS_768,
    GLOBAL_MLP_HIDDEN_DIM,
    LOCAL_MLP_HIDDEN_DIM,
    NUM_SPLANIFOLDS,
    NUM_SUBSPACES,
    ONE_TENTH_GLOBAL_MLP_HIDDEN_DIM,
    ONE_TENTH_LOCAL_MLP_HIDDEN_DIM,
    ONE_TENTH_NUM_SPLANIFOLDS,
    ONE_TENTH_NUM_SUBSPACES,
    ONE_TENTH_SUBSPACE_RANK,
    ONE_TENTH_TARGET_PARAMS_768,
    ONE_TENTH_TOPK_NEIGHBORS,
    ONE_TENTH_TOTAL_PARAMS_768,
    SSINF3Layer,
    SUBSPACE_RANK,
    TOPK_NEIGHBORS,
)

if dist.is_available() and dist.is_initialized():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
else:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton-cache/rank_{local_rank}"
os.environ["TORCHINDUCTOR_DIR"] = f"/tmp/torchinductor-cache/rank_{local_rank}"

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


TRANSFORMER_BASE_LR = 0.04
SSINF3_OPTIMIZED_BASE_LR = _env_float("SSINF3_OPTIMIZED_BASE_LR", 0.08)
SSINF3_OPTIMIZED_GROUP_LRS = {
    "global_mlp": _env_float("SSINF3_LR_GLOBAL_MLP", SSINF3_OPTIMIZED_BASE_LR),
    "local_mlp": _env_float("SSINF3_LR_LOCAL_MLP", SSINF3_OPTIMIZED_BASE_LR),
    "basis": _env_float("SSINF3_LR_BASIS", SSINF3_OPTIMIZED_BASE_LR),
    "anchors": _env_float("SSINF3_LR_ANCHORS", SSINF3_OPTIMIZED_BASE_LR),
    "tangents": _env_float("SSINF3_LR_TANGENTS", SSINF3_OPTIMIZED_BASE_LR),
    "scalars": _env_float("SSINF3_LR_SCALARS", SSINF3_OPTIMIZED_BASE_LR),
}
SSINF3_OPTIMIZED_GROUP_ORDER = (
    "global_mlp",
    "local_mlp",
    "basis",
    "anchors",
    "tangents",
    "scalars",
)


# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_k = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_v = CastedLinear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)
        self.lamb = nn.Parameter(torch.tensor(0.5)) # @Grad62304977

    def forward(self, x, v1=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        if v1 is None:
            v1 = v # This happens if we are in the first block. v needs to be accessed by subsequent blocks
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v) # @Grad62304977
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y, v1

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = CastedLinear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = CastedLinear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class SSINF3WeightMatchedMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.in_features = config.n_embd
        self.layer = SSINF3Layer(
            in_features=self.in_features,
            out_features=self.in_features,
        )

    def forward(self, x):
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = self.layer(x_flat)
        return y_flat.reshape_as(x)


class SSINF3OneTenthMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.in_features = config.n_embd
        self.layer = SSINF3Layer(
            in_features=self.in_features,
            out_features=self.in_features,
            num_subspaces=ONE_TENTH_NUM_SUBSPACES,
            subspace_rank=ONE_TENTH_SUBSPACE_RANK,
            num_splanifolds=ONE_TENTH_NUM_SPLANIFOLDS,
            local_hidden=ONE_TENTH_LOCAL_MLP_HIDDEN_DIM,
            global_hidden=ONE_TENTH_GLOBAL_MLP_HIDDEN_DIM,
            topk_neighbors=ONE_TENTH_TOPK_NEIGHBORS,
        )
        total_params = sum(p.numel() for p in self.layer.parameters())
        if total_params != ONE_TENTH_TOTAL_PARAMS_768:
            raise ValueError(
                f"invalid one-tenth SSINF3 layout: expected {ONE_TENTH_TOTAL_PARAMS_768} params, got {total_params}"
            )

    def forward(self, x):
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = self.layer(x_flat)
        return y_flat.reshape_as(x)

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(F.rms_norm(x, (x.size(-1),)), v1)
        x = x + x1
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x, v1

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768
    ssinf3_weight_matched : bool = False
    ssinf3_one_tenth : bool = False

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

        # U-net design by @brendanh0gan
        self.encoder_layers = config.n_layer // 2 # Half of the layers for encoder
        self.decoder_layers = config.n_layer - self.encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.decoder_layers))

        if config.ssinf3_weight_matched and config.ssinf3_one_tenth:
            raise ValueError("use only one of --ssinf3-weight-matched and --ssinf3-one-tenth")

        if config.ssinf3_weight_matched:
            if config.n_embd != 768:
                raise ValueError(
                    "--ssinf3-weight-matched currently supports n_embd=768 only; "
                    f"got n_embd={config.n_embd}"
                )
            for i in range(self.decoder_layers):
                block_idx = self.encoder_layers + i
                self.transformer.h[block_idx].mlp = SSINF3WeightMatchedMLP(config)
        if config.ssinf3_one_tenth:
            if config.n_embd != 768:
                raise ValueError(
                    "--ssinf3-one-tenth currently supports n_embd=768 only; "
                    f"got n_embd={config.n_embd}"
                )
            for i in range(self.decoder_layers):
                block_idx = self.encoder_layers + i
                self.transformer.h[block_idx].mlp = SSINF3OneTenthMLP(config)

        self.lm_head = CastedLinear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_() # @Grad62304977

    def forward(self, idx, target):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = F.rms_norm(x, (x.size(-1),)) # @Grad62304977
        x0 = x
        v1 = None

        # Store outputs for U-Net skip connections
        skip_connections = []

        # Encoder pass - process only the first half of the blocks
        for i in range(self.encoder_layers):
            x, v1 = self.transformer.h[i](x, v1, x0)
            skip_connections.append(x)  # Store the output for skip connections

        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.decoder_layers):
            skip_connection = skip_connections.pop()  # Get the corresponding encoder output
            # Apply learnable weight to skip connection
            weighted_skip = self.skip_weights[i] * skip_connection
            x, v1 = self.transformer.h[self.encoder_layers + i](x + weighted_skip, v1, x0)

        x = F.rms_norm(x, (x.size(-1),))
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30) # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss.float()

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 64 # batch size, in sequences, across all devices
    device_batch_size : int = 1 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 100 # number of iterations to run
    warmup_iters : int = 0
    warmdown_iters : int = 900 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 0 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 1024 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    # architecture toggles
    ssinf3_weight_matched : bool = False
    ssinf3_one_tenth : bool = False
    ssinf3_optimized : bool = False


def _parse_cli_flags():
    parser = argparse.ArgumentParser(add_help=True)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--ssinf3-weight-matched",
        action="store_true",
        help="Replace decoder MLPs with SSINF3 blocks matched to the original MLP parameter budget.",
    )
    mode_group.add_argument(
        "--ssinf3-one-tenth",
        action="store_true",
        help="Replace decoder MLPs with SSINF3 blocks sized to one-tenth of the baseline MLP parameter budget.",
    )
    parser.add_argument(
        "--ssinf3-optimized",
        action="store_true",
        help="Enable optimizer profile tuned for SSINF3 training (no model architecture changes).",
    )
    cli_args, _ = parser.parse_known_args()
    return cli_args


cli_args = _parse_cli_flags()
args = Hyperparameters(
    ssinf3_weight_matched=cli_args.ssinf3_weight_matched,
    ssinf3_one_tenth=cli_args.ssinf3_one_tenth,
    ssinf3_optimized=cli_args.ssinf3_optimized,
)

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(
    GPTConfig(
        vocab_size=num_vocab,
        n_layer=12,
        n_head=6,
        n_embd=768,
        ssinf3_weight_matched=args.ssinf3_weight_matched,
        ssinf3_one_tenth=args.ssinf3_one_tenth,
    )
)
if master_process and args.ssinf3_weight_matched:
    print(
        "using ssinf3 weight-matched decoder mlps: "
        f"subspaces={NUM_SUBSPACES} "
        f"rank={SUBSPACE_RANK} "
        f"topk={TOPK_NEIGHBORS} "
        f"splanifolds={NUM_SPLANIFOLDS} "
        f"local_hidden={LOCAL_MLP_HIDDEN_DIM} "
        f"global_hidden={GLOBAL_MLP_HIDDEN_DIM}"
    )
if master_process and args.ssinf3_one_tenth:
    print(
        "using ssinf3 one-tenth decoder mlps: "
        f"target={ONE_TENTH_TARGET_PARAMS_768:.1f} "
        f"actual={ONE_TENTH_TOTAL_PARAMS_768} "
        f"delta={ONE_TENTH_TOTAL_PARAMS_768 - ONE_TENTH_TARGET_PARAMS_768:.1f} "
        f"ratio={ONE_TENTH_TOTAL_PARAMS_768 / BASELINE_MLP_PARAMS_768:.4f} "
        f"subspaces={ONE_TENTH_NUM_SUBSPACES} "
        f"rank={ONE_TENTH_SUBSPACE_RANK} "
        f"topk={ONE_TENTH_TOPK_NEIGHBORS} "
        f"splanifolds={ONE_TENTH_NUM_SPLANIFOLDS} "
        f"local_hidden={ONE_TENTH_LOCAL_MLP_HIDDEN_DIM} "
        f"global_hidden={ONE_TENTH_GLOBAL_MLP_HIDDEN_DIM}"
    )
model = model.cuda().bfloat16()
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()

if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
# model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model

# CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
# from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
# enable_cudnn_sdp(True)
# enable_flash_sdp(False)
# enable_mem_efficient_sdp(False)
# enable_math_sdp(False)

# init the optimizer(s)
optimizer3 = None
optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=0.6,   betas=(0.9, 0.95), fused=True)
optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=0.008, betas=(0.9, 0.95), fused=True)
params = list(raw_model.transformer.h.parameters())
optimizers = [optimizer1, optimizer2]
if args.ssinf3_optimized:
    ssinf_params = [p for p in params if getattr(p, "label", None) == "ssinf"]
    core_params = [p for p in params if getattr(p, "label", None) != "ssinf"]

    matrix_params = [p for p in core_params if p.ndim == 2]
    scalar_params = [p for p in core_params if p.ndim != 2] + [raw_model.skip_weights]
    if matrix_params:
        optimizer3 = Muon(matrix_params, lr=TRANSFORMER_BASE_LR, momentum=0.95)
        optimizers.append(optimizer3)
    if scalar_params:
        # note that this learning rate is neither sensitive nor tuned
        optimizers.append(
            torch.optim.Adam(scalar_params, lr=TRANSFORMER_BASE_LR, betas=(0.9, 0.95), fused=True)
        )

    if ssinf_params:
        grouped_ssinf_params = {}
        for p in ssinf_params:
            group_name = getattr(p, "ssinf_group", "other")
            grouped_ssinf_params.setdefault(group_name, []).append(p)
        ssinf_param_groups = []
        for group_name in SSINF3_OPTIMIZED_GROUP_ORDER:
            if group_name in grouped_ssinf_params:
                ssinf_param_groups.append(
                    {
                        "params": grouped_ssinf_params.pop(group_name),
                        "lr": SSINF3_OPTIMIZED_GROUP_LRS[group_name],
                        "group_name": group_name,
                    }
                )
        for group_name in sorted(grouped_ssinf_params.keys()):
            ssinf_param_groups.append(
                {
                    "params": grouped_ssinf_params[group_name],
                    "lr": SSINF3_OPTIMIZED_BASE_LR,
                    "group_name": group_name,
                }
            )
        ssinf_optimizer = torch.optim.Adam(ssinf_param_groups, betas=(0.9, 0.95), fused=True)
        optimizers.append(ssinf_optimizer)

        if master_process:
            print("using ssinf3 optimized optimizer profile")
            for pg in ssinf_optimizer.param_groups:
                group_name = pg.get("group_name", "other")
                group_params = sum(p.numel() for p in pg["params"])
                print(f"  ssinf group={group_name} lr={pg['lr']:.4g} params={group_params}")
    elif master_process:
        print("ssinf3 optimized flag enabled but no ssinf params found; using default optimizer split")
else:
    matrix_params = [p for p in params if p.ndim == 2]
    scalar_params = [p for p in params if p.ndim != 2] + [raw_model.skip_weights]
    optimizer3 = Muon(matrix_params, lr=TRANSFORMER_BASE_LR, momentum=0.95)
    optimizers.append(optimizer3)
    # note that this learning rate is neither sensitive nor tuned
    optimizers.append(
        torch.optim.Adam(scalar_params, lr=TRANSFORMER_BASE_LR, betas=(0.9, 0.95), fused=True)
    )
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `amd-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\namd-smi:\n")
        import subprocess
        result = subprocess.run(['amd-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            with torch.no_grad():
                x_val, y_val = val_loader.next_batch()
                val_loss += model(x_val, y_val)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        loss = model(x, y)
        train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # momentum warmup for Muon
    if optimizer3 is not None:
        frac = min(step/500, 1)
        optimizer3.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
