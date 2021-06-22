#%%
from abc import abstractmethod
from argparse import ArgumentParser
from typing import Optional

from pytorch_lightning import LightningDataModule
from smart_compress.data.datasets.tiny_imagenet import TinyImageNet
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
transform = transforms.Compose(
    [
        test_transform,
    ]
)


class TinyImageNetDataModule(LightningDataModule):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--val_batch_size", type=int, help="validation batch size")
        return parser

    def __init__(self, hparams):
        super(TinyImageNetDataModule, self).__init__()

        self.hparams = hparams
        if self.hparams.val_batch_size is None:
            self.hparams.val_batch_size = max(self.hparams.batch_size // 4, 1)

    @abstractmethod
    def make_dataset(self, name: str, *args, **kwargs):
        return TinyImageNet(f"/datasets/tiny-imagenet/", *args, **kwargs)

    def setup(self, stage: Optional[str]):
        if stage == "fit" or stage is None:

            self.train_dataset = self.make_dataset(
                "train", transform=transform, split="train", download=True
            )
            self.val_dataset = self.make_dataset(
                "test", transform=test_transform, split="val", download=True
            )
        if stage == "test" or stage is None:
            self.test_dataset = self.make_dataset(
                "test", transform=test_transform, split="val", download=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
