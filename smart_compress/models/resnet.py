from argparse import ArgumentParser

import pytorch_lightning.metrics.functional.classification as FMC
from argparse_utils.mapping import mapping_action
from smart_compress.models.base import BaseModule
from smart_compress.models.pytorch.resnet import resnet18, resnet34, resnet50


class ResNetModule(BaseModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(
            parents=[BaseModule.add_argparse_args(parent_parser)], add_help=False
        )
        parser.add_argument(
            "--resnet_model",
            action=mapping_action(
                dict(resnet18=resnet18, resnet34=resnet34, resnet50=resnet50)
            ),
            default="resnet34",
            dest="resnet_model_fn",
        )
        parser.add_argument("--num_classes", default=10, type=int)
        return parser

    def __init__(self, *args, **kwargs):
        super(ResNetModule, self).__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.model = self.hparams.resnet_model_fn(num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.model(x)

    def accuracy_function(self, outputs, ground_truth):
        return dict(
            accuracy=FMC.accuracy(
                outputs, ground_truth, num_classes=self.hparams.num_classes
            )
        )
