import torch
from smart_compress.util.pytorch.quantization import float_quantize


@torch.no_grad()
def fp8_compress(x: torch.Tensor, hparams):
    return float_quantize(x, exp=5, man=2, hparams=hparams)
