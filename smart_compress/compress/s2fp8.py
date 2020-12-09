import torch
from smart_compress.util.quantization import float_quantize


def compress_fp8_squeeze_(X: torch.Tensor, args):
    X_abs = X.abs()
    X_abs_log2 = torch.where(
        X_abs == 0, X_abs, torch.log2(X_abs, out=torch.empty_like(X_abs))
    )

    mu = torch.mean(X_abs_log2)
    m = torch.max(X_abs_log2)

    alpha = 15 / (m - mu)
    beta = -alpha * mu

    return (
        float_quantize(X_abs.pow_(alpha).mul_(2 ** beta), exp=5, man=2)
        .mul_(2 ** (-beta))
        .pow_(1 / alpha)
        .mul_(torch.sign(X))
    )
