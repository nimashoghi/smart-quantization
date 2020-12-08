from smart_compress.data.cifar_base import CIFARBaseDataModule
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(CIFARBaseDataModule):
    def __init__(self, *args, **kwargs):
        super(CIFAR10DataModule, self).__init__(*args, **kwargs)

    def make_dataset(self, name, *args, **kwargs):
        return CIFAR10(f"./datasets/cifar10/cifar10-{name}", *args, **kwargs)
