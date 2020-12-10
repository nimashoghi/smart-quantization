from argparse import ArgumentParser
import torch
from qtorch.quant.quant_function import float_quantize as quantize
from torch import nn

__all__ = ["lower", "sequential_lower"]

SEQUENTIAL_LAYERS = [nn.Sequential, nn.ModuleList]  # TODO: Param List

DICT_LAYERS = [nn.ModuleDict]

CONV_LAYERS = [
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Unfold,
    nn.Fold,
]

POOL_LAYERS = [
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.MaxUnpool1d,
    nn.MaxUnpool2d,
    nn.MaxUnpool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.FractionalMaxPool2d,
    nn.LPPool1d,
    nn.LPPool2d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveMaxPool3d,
    nn.AdaptiveAvgPool3d,
]

PAD_LAYERS = [
    nn.ReflectionPad1d,
    nn.ReflectionPad2d,
    nn.ReplicationPad1d,
    nn.ReplicationPad2d,
    nn.ZeroPad2d,
    nn.ConstantPad1d,
    nn.ConstantPad2d,
    nn.ConstantPad3d,
]

ACTIVATION_LAYERS = [
    nn.ELU,
    nn.Hardshrink,
    nn.Hardtanh,
    nn.LeakyReLU,
    nn.LogSigmoid,
    nn.PReLU,
    nn.ReLU,
    nn.ReLU6,
    nn.RReLU,
    nn.SELU,
    nn.Sigmoid,
    nn.Softplus,
    nn.Softshrink,
    nn.Softsign,
    nn.Tanh,
    nn.Tanhshrink,
    nn.Threshold,
    nn.Softmin,
    nn.Softmax,
    nn.Softmax2d,
    nn.LogSoftmax,
]  # nn.AdaptiveLogSoftmaxWithLoss]

NORM_LAYERS = [
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
    nn.LocalResponseNorm,
]

# Not supporting RNN layer

LINEAR_LAYERS = [nn.Linear, nn.Bilinear]

DROPOUT_LAYERS = [nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout]

# Not supporting Sparse/Distance layers

LOSS_LAYERS = [
    nn.L1Loss,
    nn.MSELoss,
    nn.CrossEntropyLoss,
    nn.NLLLoss,
    nn.PoissonNLLLoss,
    nn.KLDivLoss,
    nn.BCELoss,
    nn.BCEWithLogitsLoss,
    nn.MarginRankingLoss,
    nn.HingeEmbeddingLoss,
    nn.MultiLabelMarginLoss,
    nn.SmoothL1Loss,
    nn.SoftMarginLoss,
    nn.MultiLabelSoftMarginLoss,  # nn.CosineEmbeddingLos,
    nn.MultiMarginLoss,
    nn.TripletMarginLoss,
]

LAYERS_TYPES = {
    "conv": CONV_LAYERS,
    "linear": LINEAR_LAYERS,
    "pool": POOL_LAYERS,
    "pad": PAD_LAYERS,
    "activation": ACTIVATION_LAYERS,
    "normalization": NORM_LAYERS,
    "dropout": DROPOUT_LAYERS,
    "loss": LOSS_LAYERS,
}


TORCH_FLOAT_MAX = torch.tensor(torch.finfo(torch.float32).max, dtype=torch.float32)
TORCH_FLOAT_EPS = torch.tensor(torch.finfo(torch.float32).eps, dtype=torch.float32)


MAX_VALUES = dict()


def _get_max_value(exp: int, man: int):
    global MAX_VALUES

    key = (exp, man)

    if key in MAX_VALUES:
        return MAX_VALUES[key]

    global TORCH_FLOAT_MAX
    MAX_VALUES[key] = max_value = quantize(
        TORCH_FLOAT_MAX, exp, man, rounding="nearest"
    )
    return max_value


def add_float_quantize_args(parent_parser: ArgumentParser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--float_quantize_check_inf", action="store_true")
    return parser


DEFAULT_LAYER_TYPES = ["conv", "linear", "pool", "normalization"]


def is_valid_layer_type(module, layer_types=DEFAULT_LAYER_TYPES):
    lp_layer_types = []
    for layer_type in layer_types:
        assert layer_type in LAYERS_TYPES.keys()
        lp_layer_types += LAYERS_TYPES[layer_type]

    return type(module) in lp_layer_types


def float_quantize(x: torch.Tensor, exp: int, man: int, hparams):
    if hparams.precision == 16:
        return_value = quantize(x.float(), exp, man, rounding="nearest")
    else:
        return_value = quantize(x, exp, man, rounding="nearest")

    if hparams.float_quantize_check_inf:
        global TORCH_FLOAT_EPS
        max_value = _get_max_value(exp, man)
        should_be_inf = torch.abs(return_value - max_value) <= TORCH_FLOAT_EPS
        return_value[should_be_inf] = float("inf")

    if hparams.precision == 16:
        return_value = return_value.half()

    return return_value
