#%%
from argparse import ArgumentParser

from smart_compress.compress.smart import SmartFP

parser = ArgumentParser()
parser = SmartFP.add_argparse_args(parser)
args = parser.parse_args([])

#%%
import torch

torch.manual_seed(100)

# %%
args.precision = 32
args.scale_factor = 1
args.use_sample_stats = True
fp = SmartFP(args)
fp

from tqdm import trange
import torch
import pandas as pd

diffs = 0

COUNT = 100

x = torch.rand((100)).random_(0, 1000)
x_orig = x.clone()

for t in trange(COUNT):
    data = []

    for i in range(1000):
        x = fp(x)
        for value, value_xorig in zip(x, x_orig):
            data.append(
                dict(
                    x_orig=float(value_xorig),
                    x=float(value),
                    x_diff=float(value - value_xorig),
                )
            )
        # print()
        # print()
        # print(x, x_orig, x - x_orig)

    df = pd.DataFrame(data)
    df["x_diff"].hist()
    m = df["x_diff"].mean()
    diffs += m
    print(m, diffs / (t + 1))

print(diffs / COUNT)

# %%
