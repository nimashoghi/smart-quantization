from abc import abstractmethod
from argparse import ArgumentParser
from enum import Enum

import pytorch_lightning as pl
import torch.nn.functional as F
from smart_compress.util.enum import ArgTypeMixin
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
        return parser

    def __init__(self, *args, optimizer_type=ArgType.SGD, **kwargs):
        super(BaseModule, self).__init__()

        self.save_hyperparameters()

    @abstractmethod
    def loss_function(self, outputs, ground_truth):
        return F.cross_entropy(outputs, ground_truth)

    @abstractmethod
    def accuracy_function(self, outputs, ground_truth):
        _, predicted = outputs.topk(5, 1, largest=True, sorted=True)
        total = ground_truth.size(0)
        ground_truth = ground_truth.view(total, -1).expand_as(predicted)
        correct = predicted.eq(ground_truth).float()
        correct_5 = correct[:, :5].sum()
        correct_1 = correct[:, :1].sum()

        return dict(correct=correct, correct_5=correct_5, correct_1=correct_1)

    def configure_optimizers(self):
        if self.hparams.optimizer_type == ArgType.SGD:
            optimizer = SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=4e-5,
                momentum=0.9,
            )
        elif self.hparams.optimizer_type == ArgType.ADAM:
            optimizer = Adam(self.parameters(), lr=self.learning_rate)
        elif self.hparams.optimizer_type == ArgType.ADAMW:
            optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        else:
            raise Exception("No optimizer")

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
