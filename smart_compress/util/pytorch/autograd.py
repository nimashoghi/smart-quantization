from argparse import Namespace

import torch
import torch.nn as nn
from smart_compress.models.base import BaseModule
from smart_compress.util.pytorch.quantization import is_valid_layer_type
from torch.autograd import Function


class Compressor(nn.Module):
    def __init__(self, compress_fn, forward=True, backward=True):
        super(Compressor, self).__init__()

        class CompressorAutoGradFn(Function):
            @staticmethod
            def forward(ctx, x: torch.Tensor):
                if not forward:
                    return x

                return compress_fn(x, tag="forward_autograd")

            @staticmethod
            def backward(ctx, grad_output):
                if not backward:
                    return grad_output

                if not ctx.needs_input_grad[0]:
                    return None

                return compress_fn(grad_output, tag="backward_autograd")

        self.compress_fn = CompressorAutoGradFn.apply

    def forward(self, x):
        return self.compress_fn(x)


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
            return compressor(module_forward(*args, **kwargs))

        module.forward = new_forward

    return model.apply(fn)
