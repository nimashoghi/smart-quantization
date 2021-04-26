from argparse import Namespace
from typing import List, Tuple

import torch
import torch.nn as nn
from smart_compress.util.pytorch.optimizer import OptimLP
from smart_compress.util.pytorch.quantization import (
    DEFAULT_LAYER_TYPES,
    is_valid_layer_type,
)
from torch.nn.modules.module import register_module_forward_hook
from torch.utils.hooks import RemovableHandle


def _wrap_fn(fn, **kwargs_fn):
    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs_fn, **kwargs)

    return wrapped


def wrap_optimizer(optimizer, compress_fn, hparams: Namespace):
    kwargs = dict()
    if hparams.compress_weights:
        kwargs["weight_quant"] = _wrap_fn(compress_fn, tag="optimizer_weight")
    if hparams.compress_gradients:
        kwargs["grad_quant"] = _wrap_fn(compress_fn, tag="optimizer_grad")
    if hparams.compress_momentum_vectors:
        kwargs["momentum_quant"] = _wrap_fn(compress_fn, tag="optimizer_momentum")

    if len(kwargs.keys()) == 0:
        return optimizer

    return OptimLP(optimizer, **kwargs)


def _register_forward_hook(compress_fn, layer_types=DEFAULT_LAYER_TYPES):
    def forward_hook(module: nn.Module, _: Tuple[torch.Tensor], output: torch.Tensor):
        if type(output) != torch.Tensor or not is_valid_layer_type(
            module, layer_types=layer_types
        ):
            return None

        return compress_fn(output, tag="forward_hook")

    return register_module_forward_hook(forward_hook)


def register_global_hooks(compress_fn, hparams, layer_types=DEFAULT_LAYER_TYPES):
    hooks: List[RemovableHandle] = []
    if hparams.compress_forward:
        hooks.append(_register_forward_hook(compress_fn, layer_types=layer_types))
    return hooks
