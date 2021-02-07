#%%
import glob
import pandas as pd


# %%
items = glob.glob(
    "/workspaces/smart-compress/lightning_logs_feb5_bert/smar*/*/metrics.csv"
)
for item in items:
    try:
        df = pd.read_csv(open(item, "r"))
        print(
            item[
                len("/workspaces/smart-compress/lightning_logs_feb5_bert/") : -len(
                    "20210206_213248/version_0/metrics.csv"
                )
            ],
            df[["val_accuracy", "val_f1"]].max(),
            {
                key: df[key].mean()
                for key in df.keys()
                if key.startswith("compression_ratio_")
            },
        )
    except:
        pass
