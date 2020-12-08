from abc import abstractmethod
from typing import Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10, CIFAR100

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transform,
    ]
)


class CIFARBaseDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(CIFARBaseDataModule, self).__init__()

        self.hparams = hparams

    @abstractmethod
    def make_dataset(self, *args, **kwargs):
        raise Exception("Not implemented")

    def setup(self, stage: Optional[str]):
        if stage == "fit" or stage is None:

            self.cifar_train = self.make_dataset(
                "train", transform=transform, train=True, download=True
            )
            self.cifar_val = self.make_dataset(
                "test", transform=test_transform, train=False, download=True
            )
        if stage == "test" or stage is None:
            self.cifar_test = self.make_dataset(
                "test", transform=test_transform, train=False, download=True
            )

    @property
    def test_batch_size(self):
        return max(self.hparams.batch_size // 4, 2)

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
