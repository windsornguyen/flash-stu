import math

import torch
import torch.nn as nn

from flash_stu.utils.numerics import nearest_power_of_two

try:
    from flash_attn import flash_attn_func as fa2
except ImportError as e:
    print(
        f"Unable to import Triton-based flash attention: {e}. No alternative currently available."
    )
    # TODO: Add FlexAttention + local attention mask when it's in stable release

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        assert torch.cuda.is_available(), "CUDA is required."
        assert config.n_embd % config.n_heads == 0
        self.n_heads = config.n_heads

        self.device = torch.device("cuda")
        self.bsz = config.bsz
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias, dtype=config.torch_dtype
        )
        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias, dtype=config.torch_dtype
        )
        self.c_proj.SCALE_INIT = 1
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(self.dropout)
        self.alibi_slopes = self._get_alibi_slopes(self.n_heads)
        self.window_size = config.window_size
        self.softcap = config.softcap

    def _generate_slopes(self, n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start**i) for i in range(n)]

    def _get_alibi_slopes(self, n_heads: int, interpolation_factor: float = 0.25):
        # If n_heads is a power of 2, generate slopes directly
        if math.log2(n_heads).is_integer():
            slopes = self._generate_slopes(n_heads)
        else:
            # Get slopes for the nearest power of two
            n = nearest_power_of_two(n_heads, round_up=False)
            slopes_power_of_two = self._generate_slopes(n)

            # Generate extra slopes
            extra_slopes = self._generate_slopes(2 * n)
            extra_slopes_trunc = extra_slopes[0::2][: n_heads - n]
            slopes = slopes_power_of_two + extra_slopes_trunc
        slopes = torch.tensor(slopes, device=self.device)
        slopes = slopes * interpolation_factor  # https://arxiv.org/pdf/2310.13017
        return slopes

    def forward(self, x):
        bsz, seq_len, d_in = x.size()

        qkv = self.c_attn(x)
        q, k, v = torch.chunk(qkv, 3, dim=2)

        q = q.view(bsz, seq_len, self.n_heads, d_in // self.n_heads)
        k = k.view(bsz, seq_len, self.n_heads, d_in // self.n_heads)
        v = v.view(bsz, seq_len, self.n_heads, d_in // self.n_heads)
        y = fa2(  # https://arxiv.org/pdf/2307.08691
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            window_size=(self.window_size, 0),
            alibi_slopes=self.alibi_slopes,  # https://arxiv.org/pdf/2108.12409
            softcap=self.softcap,  # https://arxiv.org/pdf/2408.00118
        )
        y = y.contiguous().view(bsz, seq_len, d_in)
        y = self.resid_dropout(self.c_proj(y))
        return y
