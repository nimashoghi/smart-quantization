#%%

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

SMART_FILE = "/workspaces/smart-compress/lightning_logs_success_image/smart-stochastic-[forward,backward,weight,grad,momentum]-[6-8]/version_0/"
FP32_FILE = "/workspaces/smart-compress/lightning_logs_success_image/fp32/version_0/"
BF16_FILE = "/workspaces/smart-compress/lightning_logs_success_image/bf16-[forward,backward,weight,grad,momentum]/version_0/"
FP16_FILE = "/workspaces/smart-compress/lightning_logs_success_image/fp16-[forward,backward,weight,grad,momentum]/version_0/"

#%%
import glob


def get_onefile(text):
    print(glob.escape(text) + "events.out.*")
    return glob.glob(glob.escape(text) + "events.out.*")[0]


ea_smart = event_accumulator.EventAccumulator(get_onefile(SMART_FILE))
ea_smart.Reload()

ea_fp32 = event_accumulator.EventAccumulator(get_onefile(FP32_FILE))
ea_fp32.Reload()

ea_fp16 = event_accumulator.EventAccumulator(get_onefile(FP16_FILE))
ea_fp16.Reload()

ea_bf16 = event_accumulator.EventAccumulator(get_onefile(BF16_FILE))
ea_bf16.Reload()

#%%
EAS = [
    (ea_smart, "SMFP"),
    (ea_fp32, "FP32"),
    # (ea_fp16, "FP16"),
    # (ea_bf16, "BF16"),
]
NAMES = [name for _, name in EAS]

#%%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2)
kwargs = dict(xlabel="", figsize=(5, 1.5))


def plot_scalar(scalar: str, **kwargs):
    # scalar = "val_loss_epoch"
    df = pd.DataFrame(
        dict(
            **{name: pd.DataFrame(ea.Scalars(scalar))["value"] for ea, name in EAS},
            # **{"step": pd.DataFrame(ea.Scalars(scalar)).step for ea, name in EAS}
        )
    ).reset_index()

    return df.plot(x="index", y=NAMES, **kwargs)


plot_scalar("train_loss_epoch", ax=axes[0], ylabel="Loss", ylim=(0, 2.5), **kwargs)
plot_scalar("val_loss_epoch", ax=axes[1], legend=None, ylim=(1.5, 4), **kwargs)
fig.text(0.5, 0.1, "Epochs", ha="center")

# plt.xlabel("Loss")
fig.subplots_adjust(bottom=0.3)

plt.savefig("normalization.pdf")

# #%%
# df = pd.DataFrame(ea_smart.Scalars("val_loss_epoch")).join(
#     pd.DataFrame(ea_fp32.Scalars("val_loss_epoch")), lsuffix="_smart", rsuffix="_fp32"
# )
# df = df.rename(columns=dict(value_smart="Smart FP (6, 8)", value_fp32="FP32"))


# df = pd.DataFrame(ea_smart.Scalars("train_loss_epoch")).join(
#     pd.DataFrame(ea_fp32.Scalars("train_loss_epoch")), lsuffix="_smart", rsuffix="_fp32"
# )
# df = df.rename(columns=dict(value_smart="Smart FP (6, 8)", value_fp32="FP32"))

# df.plot(
#     x="step_smart", y=["Smart FP (6, 8)", "FP32"], ax=axes[0], sharey=True, **kwargs
# )

# plt.savefig("normalization.pdf")

# %%
