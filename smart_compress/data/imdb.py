from typing import Optional

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)


class IMDBDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(IMDBDataModule, self).__init__()

        self.hparams = hparams
        self.tokenizer = self.hparams.tokenizer_cls.from_pretrained(
            self.hparams.bert_model
        )

    def batch_collate(self, batch):
        input = self.tokenizer(
            [value["text"] for value in batch],
            max_length=self.hparams.input_length,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=TruncationStrategy.LONGEST_FIRST,
            return_tensors=TensorType.PYTORCH,
        )
        output = torch.tensor([[value["label"]] for value in batch], dtype=torch.long)

        return (
            dict(
                input_ids=input["input_ids"],
                input_mask=input["attention_mask"],
            ),
            output,
        )

    def setup(self, stage: Optional[str]):
        self.dataset = load_dataset("imdb")

        if stage == "fit" or stage is None:
            self.imdb_train = self.dataset["train"]
            self.imdb_val = self.dataset["test"]
        if stage == "test" or stage is None:
            self.imdb_test = self.dataset["test"]

    @property
    def test_batch_size(self):
        return max(self.hparams.batch_size // 4, 2)

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
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.batch_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.imdb_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.batch_collate,
        )
