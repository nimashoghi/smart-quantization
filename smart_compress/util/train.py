from argparse import ArgumentParser
from enum import Enum, auto

from argparse_utils import enum_action, mapping_action
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from smart_compress.util.pytorch.autograd import register_autograd_module
from smart_compress.util.pytorch.hooks import register_global_hooks


class CompressionHookMethod(Enum):
    AUTOGRAD = auto()
    GLOBAL_HOOK = auto()


def init_model_from_args():
    from smart_compress.compress.bf16 import BF16
    from smart_compress.compress.fp8 import FP8
    from smart_compress.compress.fp16 import FP16
    from smart_compress.compress.fp32 import FP32
    from smart_compress.compress.s2fp8 import S2FP8
    from smart_compress.compress.smart import SmartFP
    from smart_compress.data.cifar10 import CIFAR10DataModule
    from smart_compress.data.cifar100 import CIFAR100DataModule
    from smart_compress.data.glue import GLUEDataModule
    from smart_compress.data.imdb import IMDBDataModule
    from smart_compress.models.bert import BertModule
    from smart_compress.models.resnet import ResNetModule

    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        action=mapping_action(dict(bert=BertModule, resnet=ResNetModule)),
        default="resnet",
        help="model name",
        dest="model_cls",
    )
    parser.add_argument(
        "--dataset",
        action=mapping_action(
            dict(
                cifar10=CIFAR10DataModule,
                cifar100=CIFAR100DataModule,
                glue=GLUEDataModule,
                imdb=IMDBDataModule,
            )
        ),
        default="cifar10",
        help="dataset name",
        dest="dataset_cls",
    )
    parser.add_argument("--no_compress", type="store_false", dest="compress")
    parser.add_argument(
        "--compress",
        action=mapping_action(
            dict(bf16=BF16, fp8=FP8, fp16=FP16, fp32=FP32, s2fp8=S2FP8, smart=SmartFP)
        ),
        default="fp32",
        dest="compression_cls",
    )
    parser.add_argument(
        "--compression_hook_method",
        action=enum_action(CompressionHookMethod, str.lower),
        default=CompressionHookMethod.AUTOGRAD,
    )
    parser.add_argument(
        "--no_compress_forward",
        action="store_false",
        dest="compress_forward",
    )
    parser.add_argument(
        "--no_compress_backward",
        action="store_false",
        dest="compress_backward",
    )
    parser.add_argument(
        "--no_compress_weights",
        action="store_false",
        dest="compress_weights",
    )
    parser.add_argument(
        "--no_compress_gradients",
        action="store_false",
        dest="compress_gradients",
    )
    parser.add_argument(
        "--no_compress_momentum_vectors",
        action="store_false",
        dest="compress_momentum_vectors",
    )
    parser.add_argument(
        "--no_compress_loss",
        action="store_false",
        dest="compress_loss",
    )
    parser.add_argument("--name", required=False, type=str)
    parser = Trainer.add_argparse_args(parser)
    args, _ = parser.parse_known_args()

    if args.model_cls == ModelType.Bert:
        assert args.dataset_cls in (GLUEDataModule, IMDBDataModule)
    elif args.model_cls == ResNetModule:
        assert args.dataset_cls in (CIFAR10DataModule, CIFAR100DataModule)
    else:
        raise Exception("invalid model_cls")

    parser = args.compression_cls.add_argparse_args(parser)
    parser = args.model_cls.add_argparse_args(parser)
    parser = args.dataset_cls.add_argparse_args(parser)

    args = parser.parse_args()
    trainer = Trainer.from_argparse_args(
        args,
        enable_pl_optimizer=True,
        logger=TensorBoardLogger("lightning_logs", name=args.name),
        terminate_on_nan=True,
    )

    compression = args.compression_cls(args)
    model = args.model_cls(compression=compression, **vars(args))
    data = args.dataset_cls(model.hparams)

    if (
        model.hparams.compress
        and model.hparams.compress != CompressionType.NoCompression
    ):
        if model.hparams.compression_hook_method == CompressionHookMethod.AUTOGRAD:
            model = register_autograd_module(model, compression, model.hparams)
        elif model.hparams.compression_hook_method == CompressionHookMethod.GLOBAL_HOOK:
            register_global_hooks(compression, model.hparams)
        else:
            raise Exception("Could not find compression_hook_method")

    return model, trainer, data
