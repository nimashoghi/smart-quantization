from abc import abstractmethod
from argparse import ArgumentParser
from enum import Enum

import pytorch_lightning as pl
import torch
from smart_compress.util.enum import ArgTypeMixin
from smart_compress.util.pytorch.hooks import wrap_optimizer
from torch.optim import SGD, Adam, AdamW


class ArgType(ArgTypeMixin, Enum):
    SGD = 0
    ADAM = 1
    ADAMW = 2


class BaseModule(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--optimizer_type",
            default=ArgType.SGD,
            type=ArgType.argtype,
            choices=ArgType,
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=0.01,
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=4e-5,
        )
        parser.add_argument(
            "--momentum",
            type=float,
            default=0.9,
        )
        return parser

    def __init__(self, *args, compress_fn=None, **kwargs):
        super(BaseModule, self).__init__()

        self.compress_fn = compress_fn
        self.save_hyperparameters()

    @abstractmethod
    def loss_function(self, outputs, ground_truth):
        raise Exception("Not implemented")

    def accuracy_function(self, outputs, ground_truth):
        _, predicted = outputs.topk(5, 1, largest=True, sorted=True)
        count = ground_truth.size(0)
        ground_truth = ground_truth.view(count, -1).expand_as(predicted)
        correct = predicted.eq(ground_truth).float()
        correct_5 = correct[:, :5].sum() / count
        correct_1 = correct[:, :1].sum() / count

        return dict(correct_5=correct_5, correct_1=correct_1)

    def configure_optimizers(self):
        if self.hparams.optimizer_type == ArgType.SGD:
            optimizer = SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer_type == ArgType.ADAM:
            optimizer = Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer_type == ArgType.ADAMW:
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise Exception("No optimizer")

        if (
            self.hparams.compress
            and (
                self.hparams.compress_weights
                or self.hparams.compress_gradients
                or self.hparams.compress_momentum_vectors
            )
            and self.compress_fn is not None
        ):

            def optimizer_compress(x: torch.Tensor):
                return self.compress_fn(x, self.hparams)

            optimizer = wrap_optimizer(optimizer, optimizer_compress, self.hparams)

        return optimizer

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, _batch_idx):
        inputs, labels = batch

        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        for metric, value in self.accuracy_function(outputs, labels).items():
            self.log(f"train_accuracy_{metric}", value, on_step=True, on_epoch=True)

        return dict(loss=loss)

    def validation_step(self, batch, _batch_idx):
        inputs, labels = batch

        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        for metric, value in self.accuracy_function(outputs, labels).items():
            self.log(f"val_accuracy_{metric}", value, on_step=True, on_epoch=True)

        return dict(loss=loss)
