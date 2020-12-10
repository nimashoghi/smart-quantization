import torch
from smart_compress.util.pytorch.quantization import float_quantize


@torch.no_grad()
def compress_fp8_squeeze(X: torch.Tensor, hparams):
    X_abs = X.abs()
    X_abs_log2 = torch.where(
        X_abs == 0, X_abs, torch.log2(X_abs, out=torch.empty_like(X_abs))
    )

    mu = torch.mean(X_abs_log2)
    m = torch.max(X_abs_log2)

    alpha = 15 / (m - mu)
    beta = -alpha * mu

    return (
        float_quantize(X_abs.pow_(alpha).mul_(2 ** beta), exp=5, man=2, hparams=hparams)
        .mul_(2 ** (-beta))
        .pow_(1 / alpha)
        .mul_(torch.sign(X))
    )
