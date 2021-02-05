import inspect
import time
from argparse import ArgumentParser
from typing import Dict, List, Union

from argparse_utils import mapping_action
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.test_tube import TestTubeLogger
from smart_compress.util.pytorch.autograd import register_autograd_module
from smart_compress.util.pytorch.hooks import register_global_hooks


def _default_name(
    args,
    data_structures=[
        "forward",
        "backward",
        "weights",
        "gradients",
        "momentum_vectors",
        "loss",
    ],
):
    tags = ",".join(
        (
            "enabled" if args.compress else "disabled",
            *(
                data_structure
                for data_structure in data_structures
                if getattr(args, f"compress_{data_structure}", False)
            ),
        )
    )

    return "-".join(
        (
            args.compression_cls.__name__,
            args.model_cls.__name__.replace("module", ""),
            args.dataset_cls.__name__.replace("datamodule", ""),
            tags,
            time.strftime("%Y%m%d_%H%M%S"),
        )
    ).lower()


def _add_arg_names(args):
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

    return args


def init_model_from_args(argv: Union[None, str, List[str]] = None):
    from smart_compress.compress.base import CompressionAlgorithmBase
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
    from smart_compress.models.base import BaseModule
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
        "--no_compress_loss",
        action="store_false",
        dest="compress_loss",
    )
    parser.add_argument("--no_add_tags", action="store_false", dest="add_tags")
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
    args = _add_arg_names(args)
    args.name = _default_name(args) if args.name is None else args.name

    trainer = Trainer.from_argparse_args(
        args,
        enable_pl_optimizer=True,
        logger=TestTubeLogger(args.logdir, name=args.name, create_git_tag=True),
        terminate_on_nan=True,
    )

    compression: CompressionAlgorithmBase = (
        args.compression_cls(args) if args.compress else None
    )
    model: BaseModule = args.model_cls(compression=compression, **vars(args))
    data = args.dataset_cls(model.hparams)

    def log_custom(metrics: Dict[str, float]):
        if not trainer.logger_connector.should_update_logs and not trainer.fast_dev_run:
            return

        trainer.logger.agg_and_log_metrics(metrics, model.global_step)

    compression.log = lambda *args, **kwargs: model.log(*args, **kwargs)
    compression.log_custom = log_custom

    if model.hparams.compress:
        model = model.hparams.compression_hook_fn(model, compression, model.hparams)

    return model, trainer, data
