#%%
from typing import List
import glob2
import pandas as pd

data: List[pd.DataFrame] = []
names: List[str] = []

LL = "lightning_logs_resnet_feb17"
for path in glob2.glob(f"/workspaces/smart-compress/{LL}/*/version_0"):
    meta = pd.read_csv(f"{path}/meta_tags.csv")
    name = meta[meta["key"] == "name"]["value"].item()
    data.append(pd.read_csv(f"{path}/metrics.csv"))
    names.append(name)

# %%
def update_name(name: str):
    l = len(
        "smartfp-resnet-cifar10-forward,backward,weights,gradients,momentum_vectors-"
    )
    if "backward" not in name:
        l -= len(",backward")
    if "momentum_vectors" not in name:
        l -= len(",momentum_vectors")
    name_updated = name[: name.find("-resnet-cifar10")] + (
        "" if "smartfp" not in name else " (" + name[l : l + 3] + ")"
    )
    name_updated = (
        name_updated.replace("smartfp", "SmaQ")
        if "smartfp" in name
        else name_updated.upper()
    )
    return name_updated + (" *" if "backward" not in name else "")


g = [
    (
        d,
        update_name(name),
    )
    for d, name in zip(data, names)
]
g = sorted(g, key=lambda x: x[1], reverse=True)
for d, name in g:
    try:
        try:
            c = d["compression_ratio"].mean()
        except:
            c = 0
        print("\t".join((str(x) for x in (name, d["val_accuracy"].max(), c))))
    except:
        pass

# %%
