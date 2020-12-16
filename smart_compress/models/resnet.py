from argparse import ArgumentParser

import torch.nn.functional as F
from argparse_utils.mapping import mapping_action
from smart_compress.models.base import BaseModule
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNetModule(BaseModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(
            parents=[BaseModule.add_argparse_args(parent_parser)], add_help=False
        )
        parser.add_argument(
            "--resnet_model",
            action=mapping_action(
                dict(
                    resnet18=resnet18,
                    resnet34=resnet34,
                    resnet50=resnet50,
                    resnet101=resnet101,
                    resnet152=resnet152,
                )
            ),
            default="resnet34",
            dest="resnet_model_fn",
        )
        parser.add_argument("--output_size", default=10, type=int)
        return parser

    def __init__(self, *args, **kwargs):
        super(ResNetModule, self).__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.model = self.hparams.resnet_model_fn()
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

        return dict(top5_accuracy=correct_5, top1_accuracy=correct_1)
