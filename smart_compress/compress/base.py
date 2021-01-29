from abc import abstractmethod
from argparse import ArgumentParser, Namespace

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

    def log_ratio(self, orig_size: float, new_size: float):
        assert hasattr(self, "log")

        if not self.hparams.measure_compression_ratio:
            return

        self.log(f"compression_ratio", orig_size / new_size, prog_bar=True)

    @abstractmethod
    def __call__(self, tensor: torch.Tensor):
        raise Exception("Not implemented")
