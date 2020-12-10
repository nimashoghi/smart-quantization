import torch
from smart_compress.util.pytorch.quantization import float_quantize


@torch.no_grad()
def bf16_compress(x: torch.Tensor, _):
    return float_quantize(x, exp=8, man=7)
