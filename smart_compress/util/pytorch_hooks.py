from argparse import Namespace
from typing import Tuple

import torch
import torch.nn as nn
from smart_compress.util.pytorch_optimizer import OptimLP
from smart_compress.util.quantization import check_layer_types
from torch.nn.modules.module import register_module_forward_hook

DEFAULT_LAYER_TYPES = ["conv", "linear", "pool", "normalization"]


def wrap_optimizer(optimizer, compress_fn, hparams: Namespace):
    kwargs = {}
    if hparams.compress_weights:
        kwargs["weight_quant"] = compress_fn
    if hparams.compress_gradients:
        kwargs["grad_quant"] = compress_fn
    if hparams.compress_momentum_vectors:
        kwargs["momentum_quant"] = compress_fn

    if len(kwargs.keys()) == 0:
        return optimizer

    return OptimLP(optimizer, **kwargs)


def register_global_hooks(compress_fn, hparams, layer_types=DEFAULT_LAYER_TYPES):
    def forward_hook(module: nn.Module, _: Tuple[torch.Tensor], output: torch.Tensor):
        if type(output) != torch.Tensor or not check_layer_types(
            module, layer_types=layer_types
        ):
            return None

        return compress_fn(output, hparams)

    register_module_forward_hook(forward_hook)
