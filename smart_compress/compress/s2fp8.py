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
    def __call__(self, tensor: torch.Tensor):
        self.log_ratio(32, 8)

        X = tensor.clone()

        X_abs = X.abs()
        X_abs_log2 = torch.where(
            X_abs == 0, X_abs, torch.log2(X_abs, out=torch.empty_like(X_abs))
        )

        mu = torch.mean(X_abs_log2)
        m = torch.max(X_abs_log2)

        alpha = 15 / (m - mu)
        beta = -alpha * mu

        return (
            float_quantize(
                X_abs.pow_(alpha).mul_(2 ** beta), exp=5, man=2, hparams=self.hparams
            )
            .mul_(2 ** (-beta))
            .pow_(1 / alpha)
            .mul_(torch.sign(X))
        )
