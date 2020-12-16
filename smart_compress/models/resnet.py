from argparse import ArgumentParser
from enum import Enum

import torch.nn.functional as F
from smart_compress.models.base import BaseModule
from smart_compress.util.enum import ArgTypeMixin
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNetModelType(ArgTypeMixin, Enum):
    resnet18 = 0
    resnet34 = 1
    resnet50 = 2
    resnet101 = 3
    resnet152 = 4


class ResNetModule(BaseModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(
            parents=[BaseModule.add_argparse_args(parent_parser)], add_help=False
        )
        parser.add_argument(
            "--resnet_model",
            default=ResNetModelType.resnet34,
            type=ResNetModelType.argtype,
            choices=ResNetModelType,
        )
        parser.add_argument("--output_size", default=10, type=int)
        return parser

    def __init__(self, *args, **kwargs):
        super(ResNetModule, self).__init__(*args, **kwargs)

        self.save_hyperparameters()

        if self.hparams.resnet_model == ResNetModelType.resnet18:
            self.model = resnet18()
        elif self.hparams.resnet_model == ResNetModelType.resnet34:
            self.model = resnet34()
        elif self.hparams.resnet_model == ResNetModelType.resnet50:
            self.model = resnet50()
        elif self.hparams.resnet_model == ResNetModelType.resnet101:
            self.model = resnet101()
        elif self.hparams.resnet_model == ResNetModelType.resnet152:
            self.model = resnet152()
        else:
            raise Exception(f"Invalid ResNet model type: {self.hparams.resnet_model}")

        modules = list(self.model.modules())
        modules[-1].out_features = self.hparams.output_size

    def forward(self, x):
        return self.model(x)

    def loss_function(self, outputs, ground_truth):
        return F.cross_entropy(outputs, ground_truth)

    def accuracy_function(self, outputs, ground_truth):
        _, predicted = outputs.topk(5, 1, largest=True, sorted=True)
        count = ground_truth.size(0)
        ground_truth = ground_truth.view(count, -1).expand_as(predicted)
        correct = predicted.eq(ground_truth).float()
        correct_5 = correct[:, :5].sum() / count
        correct_1 = correct[:, :1].sum() / count

        return dict(correct_5=correct_5, correct_1=correct_1)
