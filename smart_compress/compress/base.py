from abc import abstractmethod
from argparse import ArgumentParser, Namespace

import torch


class CompressionAlgorithmBase:
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser

    def __init__(self, hparams: Namespace):
        super(CompressionAlgorithmBase, self).__init__()

        self.hparams = hparams

    def update_hparams(self, hparams: Namespace):
        self.hparams = hparams

    @abstractmethod
    def __call__(self, tensor: torch.Tensor):
        raise Exception("Not implemented")

    def __getstate__(self):
        return (self.__class__.__name__, self.hparams)

    def __setstate__(self, value):
        _, self.hparams = value
