#%%
import torch
from torchvision.models import resnet34

model = resnet34(pretrained=True)
model

# %%
from smart_compress.data.cifar10 import CIFAR10DataModule

data = CIFAR10DataModule()
data
# %%
