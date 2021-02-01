import inspect
from argparse import ArgumentParser
from typing import List, Union

from argparse_utils import mapping_action
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from smart_compress.util.pytorch.autograd import register_autograd_module
from smart_compress.util.pytorch.hooks import register_global_hooks


def add_arg_names(args):
    for name, value in dict(**vars(args)).items():
        if value is None:
            continue

        if name.endswith("_cls"):
            assert inspect.isclass(value), f"{name} is not a class"
            setattr(
                args,
                f"{name}_name",
                value.__name__
                if value.__module__ is None
                or value.__module__ == str.__class__.__module__
                else f"{value.__module__}.{value.__name__}",
            )
        elif name.endswith("_fn"):
            assert inspect.isfunction(value), f"{name} is not a function"
            setattr(args, f"{name}_name", value.__name__)


def init_model_from_args(argv: Union[None, str, List[str]] = None):
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
    from smart_compress.models.inception import InceptionModule
    from smart_compress.models.resnet import ResNetModule

    if type(argv) == str:
        argv = argv.split(" ")

    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        action=mapping_action(
            dict(bert=BertModule, inception=InceptionModule, resnet=ResNetModule)
        ),
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
    parser.add_argument("--no_compress", action="store_false", dest="compress")
    parser.add_argument(
        "--compress",
        action=mapping_action(
            dict(bf16=BF16, fp8=FP8, fp16=FP16, fp32=FP32, s2fp8=S2FP8, smart=SmartFP)
        ),
        default="fp32",
        dest="compression_cls",
    )
    parser.add_argument(
        "--compression_hook_fn",
        action=mapping_action(
            dict(autograd=register_autograd_module, global_hook=register_global_hooks)
        ),
        default="autograd",
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
        "--compress_loss",
        action="store_true",
        dest="compress_loss",
    )
    parser.add_argument("--name", required=False, type=str)
    parser.add_argument("--logdir", default="lightning_logs", type=str)
    parser = Trainer.add_argparse_args(parser)
    args, _ = parser.parse_known_args(argv)

    if args.model_cls in (BertModule,):
        assert args.dataset_cls in (GLUEDataModule, IMDBDataModule)
    elif args.model_cls in (ResNetModule, InceptionModule):
        assert args.dataset_cls in (CIFAR10DataModule, CIFAR100DataModule)
    else:
        raise Exception("invalid model_cls")

    parser = args.compression_cls.add_argparse_args(parser)
    parser = args.model_cls.add_argparse_args(parser)
    parser = args.dataset_cls.add_argparse_args(parser)

    args = parser.parse_args(argv)
    trainer = Trainer.from_argparse_args(
        args,
        enable_pl_optimizer=True,
        logger=TensorBoardLogger(args.logdir, name=args.name),
        terminate_on_nan=True,
    )

    add_arg_names(args)

    compression = args.compression_cls(args) if args.compress else None
    model = args.model_cls(compression=compression, **vars(args))
    compression.log = lambda *args, **kwargs: model.log(*args, **kwargs)
    data = args.dataset_cls(model.hparams)

    if model.hparams.compress:
        model = model.hparams.compression_hook_fn(model, compression, model.hparams)

    return model, trainer, data
