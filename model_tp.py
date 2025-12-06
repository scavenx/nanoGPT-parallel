import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist

from model import GPT, GPTConfig, Block, LayerNorm, CausalSelfAttention, MLP


class ColumnParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, tp_group=None):
        self.tp_group = tp_group
        if tp_group is not None:
            self.tp_size = dist.get_world_size(group=tp_group)
        else:
            self.tp_size = 1
        self.out_features_local = out_features // self.tp_size

        # Initialize with the split output dimension
        super().__init__(in_features, self.out_features_local, bias=bias)

    def forward(self, x):
        input_parallel = CopyToMP.apply(x, self.tp_group)
        return super().forward(input_parallel)


class RowParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, tp_group=None):
        self.tp_group = tp_group
        if tp_group is not None:
            self.tp_size = dist.get_world_size(group=tp_group)
        else:
            self.tp_size = 1
        self.in_features_local = in_features // self.tp_size

        # Initialize with the split input dimension, bias disabled in the matmul.
        super().__init__(self.in_features_local, out_features, bias=False)

        # Add bias later manually
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        output_parallel = super().forward(x)
        output = ReduceFromMP.apply(output_parallel, self.tp_group)
        # Add back bias
        if self.bias is not None:
            output = output + self.bias
        return output

class CopyToMP(torch.autograd.Function):
    def forward(ctx, input, group):
        ctx.group = group
        return input

    def backward(ctx, grad_output):
        if ctx.group is not None:
            dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_output, None


class ReduceFromMP(torch.autograd.Function):
    def forward(ctx, input, group):
        if group is not None:
            dist.all_reduce(input, op=dist.ReduceOp.SUM, group=group)
        return input

    def backward(ctx, grad_output):
        return grad_output, None


class CausalSelfAttentionTensorParallel(CausalSelfAttention):
    def __init__(self, config, tp_group=None):
        nn.Module.__init__(self)

        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(group=tp_group) if tp_group is not None else 1

        self.n_head_local = config.n_head // self.tp_size
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn_q = ColumnParallelLinear(config.n_embd, config.n_embd, bias=config.bias, tp_group=tp_group)
        self.c_attn_k = ColumnParallelLinear(config.n_embd, config.n_embd, bias=config.bias, tp_group=tp_group)
        self.c_attn_v = ColumnParallelLinear(config.n_embd, config.n_embd, bias=config.bias, tp_group=tp_group)

        # Output projection
        self.c_proj = RowParallelLinear(config.n_embd, config.n_embd, bias=config.bias, tp_group=tp_group)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        # (B, T, C / tp_size)
        q = self.c_attn_q(x)
        k = self.c_attn_k(x)
        v = self.c_attn_v(x)

        # Reshape (B, T, n_head_local, head_size) -> (B, n_head_local, T, head_size)
        head_dim = C // (self.n_head_local * self.tp_size)
        k = k.view(B, T, self.n_head_local, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head_local, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head_local, head_dim).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # (B, nh, T, hs) -> (B, T, nh**hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C // self.tp_size)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLPTensorParallel(MLP):
    def __init__(self, config, tp_group=None):
        super().__init__(config)
        self.c_fc = ColumnParallelLinear(config.n_embd, 4 * config.n_embd, bias=config.bias, tp_group=tp_group)
        self.gelu = nn.GELU()
        self.c_proj = RowParallelLinear(4 * config.n_embd, config.n_embd, bias=config.bias, tp_group=tp_group)


class BlockTensorParallel(Block):
    def __init__(self, config, tp_group=None):
        curr_rng = torch.get_rng_state() # force same seed state globally
        torch.manual_seed(1337)
        super().__init__(config)

        torch.set_rng_state(curr_rng)

        if tp_group is not None:
            tp_rank = dist.get_rank(group=tp_group)
            torch.manual_seed(1337 + tp_rank)
        self.attn = CausalSelfAttentionTensorParallel(config, tp_group=tp_group)
        self.mlp = MLPTensorParallel(config, tp_group=tp_group)

        # Restore back
        torch.set_rng_state(curr_rng)


class GPTTensorParallel(GPT):
    def __init__(self, config, tp_group=None):
        torch.manual_seed(1337)

        super().__init__(config)
        self.tp_group = tp_group

        # weights untied
        self.transformer.wte.weight = nn.Parameter(self.transformer.wte.weight.clone())

        # Replace the blocks
        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            self.transformer.h = nn.ModuleList([BlockTensorParallel(config, tp_group) for _ in range(config.n_layer)])

            # Init weights for the new blocks
            self.apply(self._init_weights)

            # Scaled init for residuals
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))