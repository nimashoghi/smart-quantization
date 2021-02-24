from argparse import ArgumentParser, Namespace

import torch
from smart_compress.compress.base import CompressionAlgorithmBase
from smart_compress.util.pytorch.quantization import (
    add_float_quantize_args,
    float_quantize,
)


class BF16(CompressionAlgorithmBase):
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
        super(BF16, self).__init__(hparams)

    @torch.no_grad()
    def __call__(self, tensor: torch.Tensor, tag: str = None, **_):
        self.log_ratio(tag, tensor.numel(), 32, 16)

        return float_quantize(tensor, exp=8, man=7, hparams=self.hparams)
