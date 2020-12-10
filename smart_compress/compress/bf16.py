import torch
from smart_compress.util.compression import compression_function
from smart_compress.util.quantization import float_quantize


@compression_function
def bf16_compress(x: torch.Tensor, _):
    return float_quantize(x, exp=8, man=7)
