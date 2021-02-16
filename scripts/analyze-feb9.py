#%%
import glob
import pandas as pd


# %%
items = glob.glob("/workspaces/smart-compress/lightning_logs_feb9_bert/*/*/metrics.csv")
data = []
dfs = dict()
for item in items:
    df_data = pd.read_csv(item.replace("metrics.csv", "meta_tags.csv"))
    name = (
        df_data[df_data["key"] == "name"]["value"]
        .item()
        .replace(
            "smartfp-bert-glue-forward,backward,weights,gradients,momentum_vectors,loss-",
            "smfp-",
        )
    )
    if "4,6" in name:
        continue
    try:
        df = pd.read_csv(open(item, "r"))
        dfs[name] = df
        cmp_ratio = 1
        try:
            cmp_ratio = df["orig_size"].sum() / df["new_size"].sum()
        except:
            pass
        data.append(
            {
                "Name": name,
                "Train Loss": df["train_loss"].min(),
                "Val Loss": df["val_loss"].min(),
                "Accuracy": df["val_accuracy"].mean(),
                "F1": df["val_f1"].mean(),
                "Compression Ratio": cmp_ratio,
            }
        )
    except:
        pass


# %%
df = pd.DataFrame(data)
df.to_csv("./data.csv", sep="\t", index=False)

# %%
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(nrows=1, ncols=2)
kwargs = dict(xlabel="", figsize=(5, 2))


ax = None
for i, name in zip((0, 2, 5), ("FP32", "BF16", "SMFP")):
    vl = df.iloc[i].Name
    for j, (n, rolling_length, legend) in enumerate(
        zip(("train_loss", "val_loss"), (50, 10), (False, True))
    ):
        max_epoch = dfs[vl].epoch.max()
        d = dfs[vl].dropna(subset=[n])
        d[name] = d[n].rolling(rolling_length).mean()
        d["Epoch"] = np.linspace(0, max_epoch, num=len(d[name]))
        kwargs2 = dict()
        if not legend:
            kwargs2["legend"] = None
        d.plot(x="Epoch", y=name, ax=axes[j], **kwargs, **kwargs2)

plt.savefig("normalization.pdf")
fig.text(0.5, 0.1, "Epochs", ha="center")
fig.subplots_adjust(bottom=0.3, wspace=0.3)
