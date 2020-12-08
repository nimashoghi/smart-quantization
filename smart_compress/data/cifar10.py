from smart_compress.data.cifar_base import CIFARBaseDataModule
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(CIFARBaseDataModule):
    def __init__(self, *args, **kwargs):
        super(CIFAR10DataModule, self).__init__(*args, **kwargs)

        self.dataset_class = CIFAR10
