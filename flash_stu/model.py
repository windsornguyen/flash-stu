import torch
import torch.nn as nn

from transformers import PreTrainedModel

from flash_stu.modules.stu import STU
from flash_stu.modules.attention import Attention
from flash_stu.utils.numerics import nearest_power_of_two
from flash_stu.config import FlashSTUConfig
from flash_stu.layers.stu_layer import STULayer
from flash_stu.layers.attention_layer import AttentionLayer

try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm as TritonNorm
    triton_norm = True
except ImportError as e:
    print(
        f"Unable to import Triton-based RMSNorm: {e}. Falling back to PyTorch implementation."
    )
    from torch.nn import RMSNorm

    triton_norm = False


class FlashSTU(PreTrainedModel):
    config_class = FlashSTUConfig

    def __init__(self, config, phi) -> None:
        super(FlashSTU, self).__init__(config)
        self.n_layers = config.n_layers
        self.n = nearest_power_of_two(config.seq_len * 2 - 1, round_up=True)
        self.phi = phi
        self.use_approx = config.use_approx
        self.use_hankel_L = config.use_hankel_L

        # TODO: Add support for Liger-Kernel Embedding once no longer experimental
        self.tok_emb = nn.Embedding(
            config.vocab_size, config.n_embd, dtype=config.torch_dtype
        )
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            # For more complex %-split arrangements, see https://arxiv.org/pdf/2406.07887
            if layer_idx % 2 == 0:
                self.layers.append(STULayer(config, self.phi, self.n))
            else:
                self.layers.append(
                    AttentionLayer(config)
                    if config.use_attn
                    else STULayer(config, self.phi, self.n)
                )

        self.norm = (
            TritonNorm(config.n_embd)
            if triton_norm
            else RMSNorm(config.n_embd, dtype=config.torch_dtype)
        )
        # TODO: Write Issue in Liger-Kernel repo to support user-defined dtype for RMS Norm
        self.norm = self.norm.to(dtype=config.torch_dtype)
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=config.bias, dtype=config.torch_dtype
        )
        self.tok_emb.weight = self.lm_head.weight

        self.std = (config.n_embd) ** -0.5
        self.apply(self._init_weights)
        print("Model Parameter Count: %.2fM\n" % (self._get_num_params() / 1e6,))

    def forward(self, x: torch.Tensor) -> torch.tensor:
        tok_emb = self.tok_emb(x)
        x = self.dropout(tok_emb)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        y_hat = self.lm_head(x)

        return y_hat

    def _get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        if hasattr(self, "pos_emb") and self.pos_emb is not None:
            n_params -= self.pos_emb.weight.numel()
        if self.tok_emb.weight is not self.lm_head.weight:
            n_params -= self.tok_emb.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                self.std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
        elif isinstance(module, STU):
            if self.use_approx:
                torch.nn.init.xavier_normal_(module.M_inputs)
                torch.nn.init.xavier_normal_(module.M_filters)
            else:
                torch.nn.init.xavier_normal_(module.M_phi_plus)
                if not self.use_hankel_L:
                    torch.nn.init.xavier_normal_(module.M_phi_minus)
        elif isinstance(module, Attention):
            torch.nn.init.xavier_normal_(module.c_attn.weight)
            torch.nn.init.xavier_normal_(module.c_proj.weight)
            if module.c_attn.bias is not None:
                torch.nn.init.zeros_(module.c_attn.bias)
            if module.c_proj.bias is not None:
                torch.nn.init.zeros_(module.c_proj.bias)
