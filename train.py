from smart_compress.util.train import init_model_from_args

model, trainer, data = init_model_from_args()

if __name__ == "__main__":
    if trainer.auto_lr_find or trainer.auto_scale_batch_size:
        trainer.tune(model, data)
    else:
        trainer.fit(model, data)
