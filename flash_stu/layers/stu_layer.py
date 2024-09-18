import torch
import torch.nn as nn

from flash_stu.modules.stu import STU
from flash_stu.modules.swiglu import MLP

try:
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP as TritonMLP
    triton_mlp = True
except ImportError as e:
    print(
        f"Unable to import Triton-based MLP: {e}. Falling back to vanilla SwiGLU MLP instead."
    )
    triton_mlp = False

try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm as TritonNorm
    triton_norm = True
except ImportError as e:
    print(
        f"Unable to import Triton-based RMSNorm: {e}. Falling back to PyTorch implementation."
    )
    from torch.nn import RMSNorm
    triton_norm = False


class STULayer(nn.Module):
    def __init__(self, config, phi, n):
        super(STULayer, self).__init__()
        self.stu_norm = (
            TritonNorm(config.n_embd)
            if triton_norm
            else RMSNorm(config.n_embd, dtype=config.torch_dtype)
        )
        self.stu = STU(config, phi, n)
        self.mlp_norm = (
            TritonNorm(config.n_embd)
            if triton_norm
            else RMSNorm(config.n_embd, dtype=config.torch_dtype)
        )
        self.mlp = (
            TritonMLP(config) if triton_mlp else MLP(config, dtype=config.torch_dtype)
        )

        # TODO: Write Issue in Liger-Kernel repo to support user-defined dtype for MLP
        self.stu_norm = self.stu_norm.to(dtype=config.torch_dtype)
        self.mlp = self.mlp.to(dtype=config.torch_dtype)
        self.mlp_norm = self.mlp_norm.to(dtype=config.torch_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.stu(self.stu_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x
