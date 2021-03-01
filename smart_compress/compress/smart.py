from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
from smart_compress.compress.base import CompressionAlgorithmBase


class SmartFP(CompressionAlgorithmBase):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(
            parents=[CompressionAlgorithmBase.add_argparse_args(parent_parser)],
            add_help=False,
        )
        parser.add_argument(
            "--num_samples",
            type=int,
            default=16,
            help="number of samples to use for mean/std_dev calculation",
        )
        parser.add_argument(
            "--use_sample_stats",
            action="store_true",
            help="use sample mean and std for smart compression",
        )
        parser.add_argument(
            "--no_stochastic_rounding",
            action="store_false",
            help="use stochastic rounding when quantizing",
            dest="stochastic_rounding",
        )
        parser.add_argument(
            "--num_bits_main",
            type=int,
            default=6,
            help="number of bits for main data (within 1 std dev)",
        )
        parser.add_argument(
            "--num_bits_outlier",
            type=int,
            default=8,
            help="number of bits for outlier data (more than 1 std dev)",
        )
        parser.add_argument(
            "--main_std_dev_threshold",
            type=float,
            default=1.0,
            help="std dev to consider something main",
        )
        parser.add_argument(
            "--outlier_std_dev_threshold",
            type=float,
            default=2.5,
            help="max std dev for outliers (everything else is clamped to this)",
        )
        parser.add_argument("--min_size", type=int, default=8)
        return parser

    def __init__(self, hparams: Namespace):
        super(SmartFP, self).__init__(hparams)

        self.range_outlier = ((2 ** (self.hparams.num_bits_outlier - 2)) - 1) / (
            self.hparams.outlier_std_dev_threshold - self.hparams.main_std_dev_threshold
        )
        self.range_normal = (
            (2 ** (self.hparams.num_bits_main - 2)) - 1
        ) / self.hparams.main_std_dev_threshold

        self.clamped_range = (
            (1e-4, 1e4) if self.hparams.precision == 16 else (1e-38, 1e38)
        )

    def _get_sample_mean_std(self, data: torch.Tensor):
        numel = torch.tensor(data.numel(), dtype=torch.long, device=data.device)
        sample_indices = (
            torch.rand(torch.min(numel, self.hparams.num_samples), device=data.device)
            .mul(numel)
            .long()
        )
        sample = data.view(-1)[sample_indices]

        return sample.mean(), sample.std(unbiased=False)

    @torch.no_grad()
    def __call__(self, data: torch.Tensor, tag: str = None, all_positive=False, **_):
        numel = data.numel()
        orig_size = numel * 32
        if numel < self.hparams.min_size:
            self.log_ratio(tag, orig_size, 32, 32)

            return data

        mean, std_dev = (
            (data.mean(), data.std())
            if not self.hparams.use_sample_stats
            else self._get_sample_mean_std(data, self.hparams)
        )

        if std_dev == 0:  # uniform dataset
            std_dev = torch.ones_like(std_dev, device=data.device)

        data = (data - mean) / std_dev.clamp(*self.clamped_range)
        # data.sub_(mean).div_(std_dev.clamp(*self.clamped_range))
        is_outlier_higher = data > self.hparams.main_std_dev_threshold
        is_outlier_lower = data < -self.hparams.main_std_dev_threshold
        is_outlier = is_outlier_higher | is_outlier_lower

        scalars = (is_outlier_higher * -self.hparams.main_std_dev_threshold) + (
            is_outlier_lower * self.hparams.main_std_dev_threshold
        )
        ranges = torch.where(is_outlier, self.range_outlier, self.range_normal)

        data = (data + scalars) * ranges

        if self.hparams.stochastic_rounding:
            probs = torch.rand_like(data, device=data.device)
            floored_data = data.floor()
            fractions = data - floored_data

            data = floored_data + F.relu((fractions - probs) + 0.5).round()
        else:
            data = data.trunc()

        data = (data / ranges) - scalars
        data = (data * std_dev) + mean

        if all_positive:
            data = data.clamp_min(0.0)

        # if not data.isfinite().all():
        #     print(f"{tag} is not finite")

        new_size = (
            torch.sum(is_outlier) * self.hparams.num_bits_outlier
            + torch.sum(~is_outlier) * self.hparams.num_bits_main
        )
        self.log_size(tag, orig_size, new_size)

        return data
