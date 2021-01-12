from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Iterator

import pytorch_lightning as pl
from argparse_utils.mapping import mapping_action
import torch
from smart_compress.util.pytorch.hooks import wrap_optimizer
from torch import nn


def make_sgd_optimizer(parameters: Iterator[nn.Parameter], hparams: Namespace):
    from torch.optim import SGD

    return SGD(
        parameters,
        lr=hparams.learning_rate,
        momentum=hparams.momentum,
        weight_decay=hparams.weight_decay,
    )


class BaseModule(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--optimizer_type",
            action=mapping_action(dict(sgd=make_sgd_optimizer)),
            default="sgd",
            dest="make_optimizer_fn",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=0.005,
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
        parser.add_argument("--measure_average_grad_norm", action="store_true")
        return parser

    def __init__(self, *args, compression=None, **kwargs):
        super(BaseModule, self).__init__()

        self.compression = compression
        if self.compression is None:
            from smart_compress.compress.fp32 import FP32

            self.compression = FP32(self.hparams)

        self.save_hyperparameters()

        if self.hparams.measure_average_grad_norm:
            self._grads = []

    def training_epoch_end(self, *args, **kwargs):
        if not self.hparams.measure_average_grad_norm:
            return super(BaseModule, self).training_step_end(*args, **kwargs)

        try:
            avg = torch.mean(torch.tensor(self._grads))
            print(f"AVERAGE: {avg}")
        except:
            pass
        return super(BaseModule, self).training_step_end(*args, **kwargs)

    @abstractmethod
    def loss_function(self, outputs, ground_truth):
        raise Exception("Not implemented")

    def accuracy_function(self, outputs, ground_truth):
        return dict()

    def configure_optimizers(self):
        optimizer = self.hparams.make_optimizer_fn(self.parameters(), self.hparams)

        if (
            self.hparams.compress_weights
            or self.hparams.compress_gradients
            or self.hparams.compress_momentum_vectors
        ):
            optimizer = wrap_optimizer(optimizer, self.compression, self.hparams)

        return optimizer

    @abstractmethod
    def forward(self, x):
        raise Exception("Not implemented")

    def calculate_loss(self, batch):
        inputs, labels = batch

        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)

        return labels, loss, outputs

    def calculate_loss_with_compression(self, batch):
        labels, loss, outputs = self.calculate_loss(batch)

        if self.hparams.compress_loss:
            loss = self.compression(loss)

        return labels, loss, outputs

    def training_step(self, batch, _batch_idx):
        labels, loss, outputs = self.calculate_loss(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        for metric, value in self.accuracy_function(outputs, labels).items():
            self.log(
                f"train_{metric}", value, on_step=True, on_epoch=True, prog_bar=True
            )

        return dict(loss=loss)

    def validation_step(self, batch, _batch_idx):
        labels, loss, outputs = self.calculate_loss(batch)

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        for metric, value in self.accuracy_function(outputs, labels).items():
            self.log(f"val_{metric}", value, on_step=True, on_epoch=True)

        return dict(loss=loss)

    def optimizer_zero_grad(self, *args, **kwargs):
        if not self.hparams.measure_average_grad_norm:
            return super(BaseModule, self).optimizer_zero_grad(*args, **kwargs)

        norms = torch.tensor(
            [
                parameter.grad.norm()
                for parameter in self.parameters()
                if parameter.grad is not None
            ]
        )

        if len(norms):
            self._grads.append(torch.mean(norms))

        return super(BaseModule, self).optimizer_zero_grad(*args, **kwargs)
