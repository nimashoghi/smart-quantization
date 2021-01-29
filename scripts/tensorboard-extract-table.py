#%%
import glob
import pathlib

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

SCALARS = ("val_accuracy_epoch", "val_f1_epoch", "val_loss_epoch")
# SCALARS = (
#     "val_accuracy_correct_1_epoch",
#     "val_accuracy_correct_5_epoch",
#     "val_loss_epoch",
# )
# BASE_PATH = "/workspaces/smart-compress/lightning_logs_success_image"
BASE_PATH = "/workspaces/smart-compress/lightning_logs_jan12_grad_0-1"
files = glob.glob(f"{BASE_PATH}/*/*/events.out.tfevents.*")

data = {}
for events_file in files:
    p = pathlib.Path(events_file)

    ea = event_accumulator.EventAccumulator(events_file)
    ea.Reload()

    name = (
        p.parts[4]
        .replace("-[forward,backward,loss,weight,grad,momentum]", "")
        .replace("-[forward,backward,weight,grad,momentum]", "")
    )
    if name not in data:
        data[name] = dict(name=name)

    print(name)
    print("===============================")

    for scalar_name in SCALARS:
        df = pd.DataFrame(ea.Scalars(scalar_name))
        value = df["value"].min() if "loss" in scalar_name else df["value"].max()

        print(scalar_name, value)
        data[name][scalar_name] = value

    print()
    print()

#%%
df = pd.DataFrame(data.values())
df.to_csv("./data.csv", sep="\t", index=False)
df