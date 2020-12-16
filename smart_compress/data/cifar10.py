from argparse import ArgumentParser

from smart_compress.data.cifar_base import CIFARBaseDataModule
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(CIFARBaseDataModule):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(
            parents=[CIFARBaseDataModule.add_argparse_args(parent_parser)],
            add_help=False,
        )
        parser.add_argument(
            "--batch_size",
            default=8,
            type=int,
            help="batch size",
        )
        return parser

    def __init__(self, *args, **kwargs):
        super(CIFAR10DataModule, self).__init__(*args, **kwargs)

    def make_dataset(self, name, *args, **kwargs):
        return CIFAR10(f"./datasets/cifar10/cifar10-{name}", *args, **kwargs)
