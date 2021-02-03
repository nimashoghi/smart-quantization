#%%
from operator import truediv
from torchvision.models.resnet import resnet34

model = resnet34(pretrained=True)
model
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
key = "conv1.weight"
df = pd.DataFrame({key: state_dict[key].flatten().numpy()})

import seaborn as sns
import numpy as np
from scipy.stats import norm

a, b = np.histogram(df[key], bins=25, density=True)
plt.hist(df[key], bins=25, density=True)
mu, sigma = norm.fit(df[key])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, "k", linewidth=2, color="red")

plt.xlim((-0.5, 0.5))
plt.savefig("weights.pdf")

# ax = sns.histplot(df[key], kde=False)
# x_dummy = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
# ax.plot(x_dummy, norm.pdf(x_dummy, mu, sigma))

#%%
df.hist(bins=15)
