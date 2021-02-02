from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Union

import torch


class CompressionAlgorithmBase:
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--measure_compression_ratio",
            action="store_true",
            dest="measure_compression_ratio",
        )
        return parser

    def __init__(self, hparams: Namespace):
        super(CompressionAlgorithmBase, self).__init__()

        self.hparams = hparams

    def update_hparams(self, hparams: Namespace):
        self.hparams = hparams

    def log_ratio(self, tag: Union[str, None], orig_size: float, new_size: float):
        if not self.hparams.measure_compression_ratio:
            return

        assert hasattr(self, "log")

        self.log(f"compression_ratio", orig_size / new_size, prog_bar=True)
        self.log(f"compression_ratio_{tag}", orig_size / new_size)

    @abstractmethod
    def __call__(self, tensor: torch.Tensor, tag: str = None):
        raise Exception("Not implemented")
