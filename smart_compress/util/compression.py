import torch


def compression_function(f):
    def wrapped(x: torch.Tensor, *args, **kwargs):
        with torch.no_grad():
            x.data = f(x, *args, **kwargs)
            return x

    return wrapped
