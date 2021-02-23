from argparse import ArgumentParser
from typing import Optional

import datasets
from pytorch_lightning import LightningDataModule
import torch
from argparse_utils.mapping import mapping_action
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)


class GLUEDataModule(LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--max_input_length", default=512, type=int)
        parser.add_argument("--batch_size", default=8, type=int, help="batch size")
        parser.add_argument("--val_batch_size", type=int, help="validation batch size")
        parser.add_argument(
            "--task_name",
            choices=list(GLUEDataModule.task_text_field_map.keys()),
            default="sst2",
        )
        parser.add_argument(
            "--tokenizer_cls",
            action=mapping_action(dict(bert=BertTokenizer)),
            default="bert",
        )
        return parser

    def __init__(self, hparams):
        super(GLUEDataModule, self).__init__()

        self.hparams = hparams

        if self.hparams.val_batch_size is None:
            self.hparams.val_batch_size = max(self.hparams.batch_size // 4, 1)

        self.tokenizer = self.hparams.tokenizer_cls.from_pretrained(
            self.hparams.pretrained_model_name
        )
        self.text_fields = self.task_text_field_map[self.hparams.task_name]
        self.num_labels = self.glue_task_num_labels[self.hparams.task_name]

    def setup(self, stage: Optional[str]):
        self.dataset = datasets.load_dataset("glue", self.hparams.task_name)

        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset["train"]
            self.val_dataset = self.dataset["validation"]
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset["test"]

    def prepare_data(self):
        datasets.load_dataset("glue", self.hparams.task_name)
        self.hparams.tokenizer_cls.from_pretrained(self.hparams.pretrained_model_name)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=8,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            pin_memory=True,
            num_workers=8,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.val_batch_size,
            pin_memory=True,
            num_workers=8,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) == 2:
            texts_or_text_pairs = [
                tuple(element[text_field] for text_field in self.text_fields)
                for element in batch
            ]
        elif len(self.text_fields) == 1:
            [text_field] = self.text_fields
            texts_or_text_pairs = [element[text_field] for element in batch]
        else:
            raise Exception("self.text_fields must be 1 or 2")

        features = self.tokenizer(
            texts_or_text_pairs,
            max_length=self.hparams.max_input_length,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=TruncationStrategy.LONGEST_FIRST,
            return_tensors=TensorType.PYTORCH,
        )

        return features, torch.tensor(
            [element["label"] for element in batch], dtype=torch.long
        )
