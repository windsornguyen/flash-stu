import torch.nn as nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, config, dtype=None):
        # https://arxiv.org/pdf/2002.05202
        super().__init__()
        dtype = dtype if dtype is not None else config.torch_dtype
        self.hidden_size = config.n_embd
        self.intermediate_size = config.n_embd * config.mlp_scale
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.bias, dtype=dtype
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.bias, dtype=dtype
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.bias, dtype=dtype
        )
        self.dropout = nn.Dropout(
            config.dropout
        )  # TODO: Write Issue in Liger-Kernel repo to support Dropout
    
    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        outputs = self.dropout(outputs)
        return outputs
