import string
from argparse import ArgumentParser
from enum import Enum

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from smart_compress.util.enum import ArgTypeMixin


class DatasetType(ArgTypeMixin, Enum):
    CIFAR10 = 0
    CIFAR100 = 1


class ModelType(ArgTypeMixin, Enum):
    ResNet = 0


def _get_model(model_type: ModelType):
    if model_type == ModelType.ResNet:
        from smart_compress.models.resnet import ResNetModule

        return ResNetModule
    else:
        raise Exception(f"Model {model_type} not found!")


def _get_datamodule(dataset_type: DatasetType):
    if dataset_type == DatasetType.CIFAR10:
        from smart_compress.data.cifar10 import CIFAR10DataModule

        return CIFAR10DataModule
    elif dataset_type == DatasetType.CIFAR100:
        from smart_compress.data.cifar100 import CIFAR100DataModule

        return CIFAR100DataModule
    else:
        raise Exception(f"Datamodule {dataset_type} not found!")


def init_model_from_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=ModelType.argtype,
        default=ModelType.ResNet,
        choices=ModelType,
        help="model name",
        dest="model_type",
    )
    parser.add_argument(
        "--dataset",
        default=DatasetType.CIFAR10,
        choices=DatasetType,
        type=DatasetType.argtype,
        help="dataset name",
        dest="dataset_type",
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="batch size",
    )
    parser = Trainer.add_argparse_args(parser)
    args, _ = parser.parse_known_args()

    model_cls = _get_model(args.model_type)
    datamodule_cls = _get_datamodule(args.dataset_type)

    parser = model_cls.add_model_specific_args(parser)
    args = parser.parse_args()

    trainer = Trainer.from_argparse_args(args)

    model = model_cls(**vars(args))
    data = datamodule_cls(model.hparams)

    return model, trainer, data
