import torch.nn.functional as F
from argparse import ArgumentParser
from enum import Enum
from smart_compress.util.enum import ArgTypeMixin
from smart_compress.models.base import BaseModule
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


class ModelType(ArgTypeMixin, Enum):
    resnet18 = 0
    resnet34 = 1
    resnet50 = 2
    resnet101 = 3
    resnet152 = 4


class ResNetModule(BaseModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[BaseModule.add_model_specific_args(parent_parser)], add_help=False
        )
        parser.add_argument(
            "--resnet_model",
            default=ModelType.resnet34,
            type=ModelType.argtype,
            choices=ModelType,
        )
        parser.add_argument("--output_size", default=10, type=int)
        return parser

    def __init__(self, *args, **kwargs):
        super(ResNetModule, self).__init__(*args, **kwargs)

        self.save_hyperparameters()

        if self.hparams.resnet_model == ModelType.resnet18:
            self.model = resnet18()
        elif self.hparams.resnet_model == ModelType.resnet34:
            self.model = resnet34()
        elif self.hparams.resnet_model == ModelType.resnet50:
            self.model = resnet50()
        elif self.hparams.resnet_model == ModelType.resnet101:
            self.model = resnet101()
        elif self.hparams.resnet_model == ModelType.resnet152:
            self.model = resnet152()
        else:
            raise Exception(f"Invalid ResNet model type: {self.hparams.resnet_model}")

        modules = list(self.model.modules())
        modules[-1].out_features = self.hparams.output_size

    def loss_function(self, outputs, ground_truth):
        return F.cross_entropy(outputs, ground_truth)
