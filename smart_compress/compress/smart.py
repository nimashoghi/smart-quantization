from argparse import ArgumentParser, Namespace

import torch


def add_args_smart_compress(parent_parser: ArgumentParser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=25,
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
    return parser


def _get_sample_mean_std(data: torch.Tensor, hparams: Namespace):
    numel = torch.tensor(data.numel(), dtype=torch.long)
    sample_indices = torch.rand(torch.min(numel, hparams.num_samples)).mul(numel).long()
    sample = data.view(-1)[sample_indices]

    return sample.mean(), sample.std(unbiased=False)


@torch.no_grad()
def compress_smart(data: torch.Tensor, hparams: Namespace):
    data = data.clone()

    mean, std_dev = (
        (data.mean(), data.std())
        if not hparams.use_sample_stats
        else _get_sample_mean_std(data, hparams)
    )

    clamped_range = (1e-4, 1e4) if hparams.precision == 16 else (1e-38, 1e38)
    std_dev.clamp_(*clamped_range)

    data.sub_(mean).div_(std_dev)
    is_outlier_higher = data > hparams.main_std_dev_threshold
    is_outlier_lower = data < -hparams.main_std_dev_threshold
    is_outlier = is_outlier_higher | is_outlier_lower

    scalars = (is_outlier_higher * -hparams.main_std_dev_threshold) + (
        is_outlier_lower * hparams.main_std_dev_threshold
    )
    ranges = torch.where(
        is_outlier,
        ((2 ** (hparams.num_bits_outlier - 2)) - 1)  # -1 for tag, -1 for sign
        / (hparams.outlier_std_dev_threshold - hparams.main_std_dev_threshold),
        ((2 ** (hparams.num_bits_main - 2)) - 1) / hparams.main_std_dev_threshold,
    )
    data.add_(scalars).mul_(ranges)

    if hparams.stochastic_rounding:
        data[(data - torch.floor(data)) >= torch.rand_like(data)] += 1
        data.floor_()
    else:
        data.trunc_()

    data.div_(ranges).sub_(scalars)
    data.mul_(std_dev).add_(mean)

    return data
