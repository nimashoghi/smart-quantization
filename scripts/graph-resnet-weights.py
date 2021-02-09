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
img, label = next(iter(dataloader))
img

#%%
output_saved = None
grad_input_saved = None
grad_output_saved = None


def fn(module, input, output):
    global output_saved

    output_saved = output


def fn_grad(module, grad_input, grad_output):
    global grad_input_saved, grad_output_saved

    grad_input_saved = grad_input
    grad_output_saved = grad_output


model.train()
handle = model.conv1.register_forward_hook(fn)
handle2 = model.conv1.register_backward_hook(fn_grad)
x = model(img)

#%%
import torch.nn.functional as F

loss = F.cross_entropy(x, label)
loss.backward()

handle.remove()
handle2.remove()

# %%
state_dict = model.state_dict()
for keys in state_dict.keys():
    if "conv" in keys:
        print(keys)
#%%
import pandas as pd

import matplotlib.pyplot as plt

for key in state_dict.keys():
    if "conv" not in key:
        continue
    df = pd.DataFrame({key: state_dict[key].flatten().numpy()})
    df.hist()


# %%

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
all_lims = (None, (-0.5, 0.5), (-0.02, 0.02), (-0.005, 0.005))

plt.subplots_adjust(hspace=0.5)

for data, ax, title, lims in zip(datas, axs.flatten(), labels, all_lims):
    ax.set_title(title)
    # fig = plt.gcf()
    # fig.set_size_inches(18.5, 5)
    a, b = np.histogram(data, bins=100, density=True)
    ax.hist(data, bins=100, density=True)
    mu, sigma = norm.fit(data)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    ax.plot(x, p, "k", linewidth=2, color="red")
    if lims is not None:
        ax.set_xlim(*lims)
    ax.get_yaxis().set_visible(False)

# plt.xlim((-0.5, 0.5))
plt.savefig("weights.pdf")

# ax = sns.histplot(df[key], kde=False)
# x_dummy = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
# ax.plot(x_dummy, norm.pdf(x_dummy, mu, sigma))

# %%
