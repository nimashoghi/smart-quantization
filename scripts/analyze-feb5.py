#%%
import glob
import pandas as pd


# %%
items = glob.glob("/workspaces/smart-compress/lightning_logs_feb5_bert/*/*/metrics.csv")
i = []
for item in items:
    try:
        df = pd.read_csv(open(item, "r"))
        i.append(df)
    except:
        i.append(None)

# %%
bf16 = i[0]
s2fp8 = i[1]
smfp_57 = i[2]
smfp_68 = i[3]
fp16 = i[5]
fp8 = i[6]
fp32 = i[7]

dfs = dict(
    bf16=bf16,
    s2fp8=s2fp8,
    smfp_57=smfp_57,
    smfp_68=smfp_68,
    fp16=fp16,
    fp8=fp8,
    fp32=fp32,
)

# %%
for key, value in dfs.items():
    print(key, None if value is None else value["val_accuracy"].max())
