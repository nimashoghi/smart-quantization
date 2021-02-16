#%%
from smart_compress.models.resnet import resnet34

model = resnet34(pretrained=True)
model

#%%
from smart_compress.data.cifar10 import CIFAR10DataModule
from argparse import Namespace

hparams = Namespace(batch_size=1, val_batch_size=1)

datamodule = CIFAR10DataModule(hparams)
datamodule.setup("fit")
dataloader = datamodule.train_dataloader()


#%%
d = iter(dataloader)

#%%

img, label = next(d)
img

#%%
output_saved = None
grad_input_saved = None
grad_output_saved = None


def fn(module, input, output):
    global output_saved

    if output_saved is None:
        output_saved = output


model.train()
handle = model.conv1.register_forward_hook(fn)
x = model(img)
handle.remove()

#%%
import torch.nn.functional as F


def fn_grad(module, grad_input, grad_output):
    global grad_input_saved, grad_output_saved

    if grad_input_saved is None:
        grad_input_saved = grad_input
    if grad_output_saved is None:
        grad_output_saved = grad_output


model = resnet34(pretrained=False)
handle2 = model.conv1.register_backward_hook(fn_grad)
x = model(img)
loss = F.cross_entropy(x, label)
loss.backward()

handle2.remove()

# %%
state_dict = model.state_dict()
for keys in state_dict.keys():
    if "conv" in keys:
        print(keys)
#%%
# import pandas as pd


# for key in state_dict.keys():
#     if "conv" not in key:
#         continue
#     df = pd.DataFrame({key: state_dict[key].flatten().numpy()})
#     df.hist()


# %%

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm

fig, axs = plt.subplots(2, 2)
datas = [
    x.flatten().detach().numpy()
    for x in [
        state_dict["conv1.weight"],
        output_saved,
        grad_input_saved[1],
        grad_output_saved[0],
    ]
]
labels = ("(a)", "(b)", "(c)", "(d)")
all_lims = (None, (-0.5, 0.5), (-25, 20), (-1, 1))
all_nbins = (None, None, None, None)

plt.subplots_adjust(hspace=0.5)

for data, ax, title, lims, nbins in zip(
    datas, axs.flatten(), labels, all_lims, all_nbins
):
    nbins = nbins or 100
    ax.set_title(title)
    # fig = plt.gcf()
    # fig.set_size_inches(18.5, 5)
    a, b = np.histogram(data, bins=nbins, density=True)
    ax.hist(data, bins=nbins, density=True)
    mu, sigma = norm.fit(data)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, nbins)
    p = norm.pdf(x, mu, sigma)
    ax.plot(x, p, "k", linewidth=2, color="red")
    if lims is not None:
        ax.set_xlim(*lims)
    ax.get_yaxis().set_visible(False)
    print(title, mu, sigma)

# plt.xlim((-0.5, 0.5))
plt.savefig("weights.pdf")

# ax = sns.histplot(df[key], kde=False)
# x_dummy = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
# ax.plot(x_dummy, norm.pdf(x_dummy, mu, sigma))
