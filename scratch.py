#%%
from smart_compress.util.train import init_model_from_args

model, trainer, data = init_model_from_args(
    '--accelerator ddp --limit_train_batches 0.5 --limit_val_batches 0.5 --gpus -1 --model bert --dataset glue --task_name mnli --batch_size 10 --val_batch_size 2 --max_epochs 5 --compress smart --num_bits_main 6 --num_bits_outlier 8 --tags "6,8" --measure_compression_ratio'
)

# %%
data.setup("fit")
