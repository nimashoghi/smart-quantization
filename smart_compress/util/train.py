from argparse import ArgumentParser
from enum import Enum
from smart_compress.util.pytorch_hooks import register_global_hooks

from pytorch_lightning import Trainer
from smart_compress.util.enum import ArgTypeMixin


class DatasetType(ArgTypeMixin, Enum):
    CIFAR10 = 0
    CIFAR100 = 1


class ModelType(ArgTypeMixin, Enum):
    ResNet = 0


class CompressionType(ArgTypeMixin, Enum):
    NoCompression = 0
    FP8 = 1
    SmartCompress = 2
    S2FP8 = 3
    FP16 = 4
    BF16 = 5
    FP32 = 6


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


def _get_compression(compression_type: CompressionType):
    if compression_type == CompressionType.FP8:
        pass
    elif compression_type == CompressionType.SmartCompress:
        from smart_compress.compress.smart import (
            add_args_smart_compress,
            compress_smart_,
        )

        return compress_smart_, add_args_smart_compress
    elif compression_type == CompressionType.S2FP8:
        from smart_compress.compress.s2fp8 import compress_fp8_squeeze_

        return compress_fp8_squeeze_, (lambda parent_parser: parent_parser)
    else:
        return (lambda x, _: x), (lambda parent_parser: parent_parser)


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
    parser.add_argument(
        "--compress",
        default=CompressionType.NoCompression,
        choices=CompressionType,
        type=CompressionType.argtype,
    )
    parser.add_argument("--compress_optimizer", action="store_true")
    parser = Trainer.add_argparse_args(parser)
    args, _ = parser.parse_known_args()

    model_cls = _get_model(args.model_type)
    datamodule_cls = _get_datamodule(args.dataset_type)
    compress_fn, add_args_fn = _get_compression(args.compress)

    parser = model_cls.add_model_specific_args(parser)
    parser = add_args_fn(parser)

    args = parser.parse_args()
    trainer = Trainer.from_argparse_args(args)

    model = model_cls(compress_fn=compress_fn, **vars(args))
    data = datamodule_cls(model.hparams)

    if args.compress and args.compress != CompressionType.NoCompression:
        register_global_hooks(compress_fn)

    return model, trainer, data
