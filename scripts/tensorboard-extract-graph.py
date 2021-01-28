#%%

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

SMART_FILE = "/workspaces/smart-compress/lightning_logs_success_image/smart-stochastic-[forward,backward,weight,grad,momentum]-[6-8]/version_0/events.out.tfevents.1607634687.hummer.cc.gt.atl.ga.us.23686.0"
FP32_FILE = "/workspaces/smart-compress/lightning_logs_success_image/fp32/version_0/events.out.tfevents.1607632699.hummer.cc.gt.atl.ga.us.4929.0"
BF16_FILE = "/workspaces/smart-compress/lightning_logs_success_image/bf16-[forward,backward,weight,grad,momentum]/version_0/events.out.tfevents.1607637453.hummer.cc.gt.atl.ga.us.58412.0"
FP16_FILE = "/workspaces/smart-compress/lightning_logs_success_image/fp16-[forward,backward,weight,grad,momentum]/version_0/events.out.tfevents.1607636568.hummer.cc.gt.atl.ga.us.43069.0"

#%%
ea_smart = event_accumulator.EventAccumulator(SMART_FILE)
ea_smart.Reload()

ea_fp32 = event_accumulator.EventAccumulator(FP32_FILE)
ea_fp32.Reload()

ea_fp16 = event_accumulator.EventAccumulator(FP16_FILE)
ea_fp16.Reload()

ea_bf16 = event_accumulator.EventAccumulator(BF16_FILE)
ea_bf16.Reload()

#%%
EAS = [
    (ea_smart, "Smart FP (6, 8)"),
    (ea_fp32, "FP32"),
    # (ea_fp16, "FP16"),
    (ea_bf16, "BF16"),
]
NAMES = [name for _, name in EAS]

#%%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2)
kwargs = dict(xlabel="Time", xticks=[], sharey=True, ylim=(0, 5), ylabel="Loss")


def plot_scalar(scalar: str, **kwargs):
    # scalar = "val_loss_epoch"
    df = pd.DataFrame(
        {name: pd.DataFrame(ea.Scalars(scalar))["value"] for ea, name in EAS}
    ).reset_index()

    return df.plot(x="index", y=NAMES, **kwargs)


plot_scalar("train_loss_epoch", ax=axes[0], legend=None, **kwargs)
plot_scalar("val_loss_epoch", ax=axes[1], **kwargs)
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
