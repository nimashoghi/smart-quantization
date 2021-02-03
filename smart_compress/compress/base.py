from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Union

import torch


@torch.no_grad()
def _reduce_fx(tensors: List[torch.Tensor]):
    if isinstance(tensors, list):
        tensors = torch.tensor(tensors, dtype=float)
    return torch.sum(tensors)


class CompressionAlgorithmBase:
    log = None
    log_custom = None

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

    def _log_scalars(self, scalars: Dict[str, float], custom=False):
        if custom:
            self.log_custom(scalars)
            return

        for name, value in scalars.items():
            kwargs = dict()
            if "size" in name:
                kwargs["reduce_fx"] = _reduce_fx
                kwargs["tbptt_reduce_fx"] = _reduce_fx

            self.log(name, value, **kwargs)

    def log_ratio(
        self,
        tag: Union[str, None],
        size: int,
        orig_bitcount: float,
        new_bitcount: float,
        overhead=0,
    ):
        return self.log_size(
            tag, size * orig_bitcount, size * new_bitcount, overhead=overhead
        )

    def log_size(
        self,
        tag: Union[str, None],
        orig_size: float,
        new_size: float,
        overhead=0,
    ):
        if not self.hparams.measure_compression_ratio:
            return

        assert hasattr(self, "log")

        new_size += overhead
        compression_ratio = orig_size / new_size

        self._log_scalars(
            {
                f"compression_ratio": compression_ratio,
                f"compression_ratio_{tag}": compression_ratio,
                f"new_size": new_size,
                f"new_size_{tag}": new_size,
                f"orig_size": orig_size,
                f"orig_size_{tag}": orig_size,
            },
            custom=tag.startswith("optimizer_"),
        )

    @abstractmethod
    def __call__(self, tensor: torch.Tensor, tag: str = None):
        raise Exception("Not implemented")
