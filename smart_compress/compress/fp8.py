import torch
from smart_compress.util.pytorch.compression import compression_function
from smart_compress.util.pytorch.quantization import float_quantize


@compression_function
def fp8_compress(x: torch.Tensor, _):
    return float_quantize(x, exp=5, man=2)
