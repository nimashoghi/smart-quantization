#%%
from argparse import ArgumentParser
from typing import Optional

import torch
from argparse_utils.mapping import mapping_action
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizer
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)


class IMDBDataModule(LightningDataModule):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--max_input_length", default=512, type=int)
        parser.add_argument("--batch_size", default=8, type=int, help="batch size")
        parser.add_argument("--val_batch_size", type=int, help="validation batch size")
        parser.add_argument(
            "--tokenizer_cls",
            action=mapping_action(dict(bert=BertTokenizer)),
            default="bert",
        )
        return parser

    def __init__(self, hparams):
        super(IMDBDataModule, self).__init__()

        self.hparams = hparams
        if self.hparams.val_batch_size is None:
            self.hparams.val_batch_size = max(self.hparams.batch_size // 4, 1)

        self.tokenizer = self.hparams.tokenizer_cls.from_pretrained(
            self.hparams.pretrained_model_name
        )

    def batch_collate(self, batch):
        input = self.tokenizer(
            [value["text"] for value in batch],
            max_length=self.hparams.max_input_length,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=TruncationStrategy.LONGEST_FIRST,
            return_tensors=TensorType.PYTORCH,
        )
        output = torch.tensor([value["label"] for value in batch], dtype=torch.long)

        return (
            dict(
                input_ids=input["input_ids"],
                attention_mask=input["attention_mask"],
            ),
            output,
            batch,
        )

    def setup(self, stage: Optional[str]):
        self.dataset = load_dataset("imdb")

        if stage == "fit" or stage is None:
            self.imdb_train = self.dataset["train"]
            self.imdb_val = self.dataset["test"]
        if stage == "test" or stage is None:
            self.imdb_test = self.dataset["test"]

    def prepare_data(self):
        load_dataset("imdb")
        self.hparams.tokenizer_cls.from_pretrained(self.hparams.pretrained_model_name)

    def train_dataloader(self):
        return DataLoader(
            self.imdb_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.batch_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.imdb_val,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.batch_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.imdb_test,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.batch_collate,
        )


# %%
from argparse import Namespace
from transformers.models.bert import BertTokenizer


hparams = Namespace(
    batch_size=1,
    val_batch_size=1,
    tokenizer_cls=BertTokenizer,
    pretrained_model_name="bert-base-uncased",
    max_input_length=512,
)
datamodule = IMDBDataModule(hparams)
datamodule.setup("fit")
dl = datamodule.train_dataloader()

# %%
next(iter(dl))
