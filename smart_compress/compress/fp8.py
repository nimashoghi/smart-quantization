import torch
from smart_compress.util.quantization import float_quantize


def fp8_compress(x: torch.Tensor, _):
    return float_quantize(x, exp=5, man=2)
