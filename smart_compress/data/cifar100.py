from smart_compress.data.cifar_base import CIFARBaseDataModule
from torchvision.datasets import CIFAR100


class CIFAR100DataModule(CIFARBaseDataModule):
    def __init__(self, *args, **kwargs):
        super(CIFAR100DataModule, self).__init__(*args, **kwargs)

    def make_dataset(self, name, *args, **kwargs):
        return CIFAR100(f"./datasets/cifar100/cifar100-{name}", *args, **kwargs)
