from argparse import ArgumentParser, Namespace

import torch
from smart_compress.compress.base import CompressionAlgorithmBase
from smart_compress.util.pytorch.quantization import (
    add_float_quantize_args,
    float_quantize,
)


class S2FP8(CompressionAlgorithmBase):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(
            parents=[
                add_float_quantize_args(
                    CompressionAlgorithmBase.add_argparse_args(parent_parser)
                )
            ],
            add_help=False,
        )
        return parser

    def __init__(self, hparams: Namespace):
        super(S2FP8, self).__init__(hparams)

    @torch.no_grad()
    def __call__(self, tensor: torch.Tensor, tag: str = None, **_):
        self.log_ratio(tag, tensor.numel(), 32, 8, overhead=64)

        X = tensor
        signs = torch.sign(X)

        X_abs = X.abs()
        X_abs_log2 = torch.where(X_abs == 0.0, X_abs, torch.log2(X_abs))

        mu = torch.mean(X_abs_log2)
        m = torch.max(X_abs_log2)

        alpha = 15.0 / (m - mu)
        beta = -alpha * mu

        beta_pow2 = 2.0 ** beta

        truncated = float_quantize(
            X_abs.pow_(alpha).mul_(beta_pow2), exp=5, man=2, hparams=self.hparams
        )
        return ((truncated * beta_pow2.reciprocal_()) ** alpha.reciprocal_()) * signs
