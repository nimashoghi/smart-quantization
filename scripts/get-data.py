#%%
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator(
    "/workspaces/smart-compress/lightning_logs_feb4_bert/smart-stochastic-[forward,backward,weight,grad,momentum]-[6-8]/version_0/events.out.tfevents.1612460866.hummer.cc.gt.atl.ga.us.49363.0"
)
ea
# %%
ea.Reload()
ea.Tags().keys()

#%%
df = pd.DataFrame()
