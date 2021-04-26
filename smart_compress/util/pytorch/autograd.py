from argparse import Namespace
from typing import List

import torch
import torch.nn as nn
from smart_compress.models.base import BaseModule
from smart_compress.util.pytorch.quantization import is_valid_layer_type
from torch._C import AnyType
from torch.autograd import Function


def process_input(args: List[AnyType]):
    if len(args) >= 1 and type(args[-1]) == dict and "batch_norm_stats" in args[-1]:
        return args[:-1], args[-1]
    return (args, {})


class Compressor(nn.Module):
    def __init__(self, compress_fn, forward=True, backward=True):
        super(Compressor, self).__init__()

        class CompressorAutoGradFn(Function):
            @staticmethod
            def forward(ctx, x: torch.Tensor, *args, **kwargs):
                if not forward:
                    return x

                args, new_kwargs = process_input(args)

                return compress_fn(
                    x, *args, **kwargs, **new_kwargs, tag="forward_autograd"
                )

            @staticmethod
            def backward(ctx, grad_output):
                if not backward:
                    return grad_output, None

                if not ctx.needs_input_grad[0]:
                    return None, None

                return compress_fn(grad_output, tag="backward_autograd"), None

        self.compress_fn = CompressorAutoGradFn.apply

    def forward(self, *args, **kwargs):
        return self.compress_fn(*args, **kwargs)


def register_autograd_module(model: BaseModule, compress_fn, hparams: Namespace):
    compressor = Compressor(
        compress_fn,
        forward=hparams.compress_forward,
        backward=hparams.compress_backward,
    )

    def fn(module: nn.Module):
        if not is_valid_layer_type(module):
            return

        module_forward = module.forward

        def new_forward(*args, **kwargs):
            if hparams.use_batch_norm and type(module) == nn.BatchNorm2d:
                new_kwargs = dict(
                    batch_norm_stats=(module.weight.detach(), module.bias.detach())
                )
                return compressor(module_forward(*args, **kwargs), new_kwargs)
            return compressor(module_forward(*args, **kwargs))

        module.forward = new_forward

    return model.apply(fn)
