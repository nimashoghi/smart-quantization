from argparse import ArgumentParser

import torchmetrics.functional as FM
from smart_compress.models.base import BaseModule
from smart_compress.models.pytorch.inception import inception_v3


class InceptionModule(BaseModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(
            parents=[BaseModule.add_argparse_args(parent_parser)], add_help=False
        )
        parser.add_argument("--num_classes", default=10, type=int)
        return parser

    def __init__(self, *args, **kwargs):
        super(InceptionModule, self).__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.model = inception_v3(num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.model(x)

    def accuracy_function(self, outputs, ground_truth):
        return dict(
            accuracy=FM.accuracy(
                outputs.argmax(dim=1),
                ground_truth,
                num_classes=self.hparams.num_classes,
            )
        )
