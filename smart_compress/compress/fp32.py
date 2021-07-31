from argparse import ArgumentParser, Namespace

import torch
from smart_compress.compress.base import CompressionAlgorithmBase


class FP32(CompressionAlgorithmBase):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(
            parents=[CompressionAlgorithmBase.add_argparse_args(parent_parser)],
            add_help=False,
        )
        return parser

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)

    @torch.no_grad()
    def __call__(self, tensor: torch.Tensor, tag: str = None, **_):
        self.log_ratio(tag, tensor.numel(), 32, 32)

        return tensor
