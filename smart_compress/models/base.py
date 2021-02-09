from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Iterator

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from argparse_utils.mapping import mapping_action
from smart_compress.util.pytorch.hooks import wrap_optimizer
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer


def make_adam_optimizer(parameters: Iterator[nn.Parameter], hparams: Namespace):
    from torch.optim import Adam

    beta_args = (
        dict(betas=(hparams.beta1, hparams.beta2))
        if hparams.beta1 and hparams.beta2
        else dict()
    )

    return Adam(
        parameters,
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay,
        **beta_args,
    )


def make_sgd_optimizer(parameters: Iterator[nn.Parameter], hparams: Namespace):
    from torch.optim import SGD

    return SGD(
        parameters,
        lr=hparams.learning_rate,
        momentum=hparams.momentum,
        weight_decay=hparams.weight_decay,
    )


def make_multistep_scheduler(optimizer: Optimizer, hparams: Namespace):
    return MultiStepLR(
        optimizer,
        milestones=hparams.scheduler_milestones,
        gamma=hparams.scheduler_gamma,
    )


class BaseModule(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--optimizer_type",
            action=mapping_action(
                dict(adam=make_adam_optimizer, sgd=make_sgd_optimizer)
            ),
            default="sgd",
            dest="make_optimizer_fn",
        )
        parser.add_argument(
            "--scheduler_type",
            action=mapping_action(dict(multi_step=make_multistep_scheduler)),
            dest="make_scheduler_fn",
        ),
        parser.add_argument("--scheduler_gamma", type=float, default=0.1)
        parser.add_argument(
            "--scheduler_milestones",
            type=int,
            nargs="+",
            default=[100, 150, 200],
        )
        parser.add_argument("--learning_rate", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--beta1", type=float)
        parser.add_argument("--beta2", type=float)
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
            return super(BaseModule, self).training_epoch_end(*args, **kwargs)

        try:
            avg = torch.mean(torch.tensor(self._grads))
            print(f"AVERAGE: {avg}")
        except:
            pass
        return super(BaseModule, self).training_epoch_end(*args, **kwargs)

    def loss_function(self, outputs, ground_truth):
        return F.cross_entropy(outputs, ground_truth)

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

        if self.hparams.make_scheduler_fn:
            scheduler = self.hparams.make_scheduler_fn(optimizer, self.hparams)
            return [optimizer], [scheduler]

        return optimizer

    @abstractmethod
    def forward(self, x):
        raise Exception("Not implemented")

    def calculate_loss(self, batch):
        inputs, labels = batch

        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)

        if self.hparams.compress_loss:
            loss.data = self.compression(loss.data, tag="loss")

        return labels, loss, outputs

    def training_step(self, batch, _batch_idx):
        labels, loss, outputs = self.calculate_loss(batch)

        self.log("train_loss", loss)
        for metric, value in self.accuracy_function(outputs, labels).items():
            self.log(f"train_{metric}", value, on_epoch=True, prog_bar=True)

        return dict(loss=loss)

    def validation_step(self, batch, _batch_idx):
        labels, loss, outputs = self.calculate_loss(batch)

        self.log("val_loss", loss)
        for metric, value in self.accuracy_function(outputs, labels).items():
            self.log(f"val_{metric}", value, on_epoch=True, prog_bar=True)

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
