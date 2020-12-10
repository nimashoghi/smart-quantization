import torch
from smart_compress.util.pytorch_hooks import compression_function
from smart_compress.util.quantization import float_quantize


@compression_function
def fp16_compress(x: torch.Tensor, _):
    return float_quantize(x, exp=5, man=10)
