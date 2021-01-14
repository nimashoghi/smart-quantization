from argparse import ArgumentParser, Namespace

import torch
from smart_compress.compress.base import CompressionAlgorithmBase
from smart_compress.util.pytorch.quantization import (
    add_float_quantize_args,
    float_quantize,
)


class FP8(CompressionAlgorithmBase):
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
        super(FP8, self).__init__(hparams)

    @torch.no_grad()
    def __call__(self, tensor: torch.Tensor):
        self.log_ratio(32, 8)
        return float_quantize(tensor, exp=5, man=2, hparams=self.hparams)
