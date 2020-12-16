from argparse import ArgumentParser

import datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)


class GLUEDataModule(pl.LightningDataModule):
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
        parser.add_argument(
            "--task_name",
            choices=list(GLUEDataModule.task_text_field_map.keys()),
            default="mrpc",
        )
        return parser

    def __init__(self, hparams):
        super(GLUEDataModule, self).__init__()

        self.hparams = hparams

        self.tokenizer = self.hparams.tokenizer_cls.from_pretrained(
            self.hparams.bert_model
        )
        self.text_fields = self.task_text_field_map[self.hparams.task_name]
        self.num_labels = self.glue_task_num_labels[self.hparams.task_name]

    def setup(self, stage):
        self.dataset = datasets.load_dataset("glue", self.hparams.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [
                c for c in self.dataset[split].column_names if c in self.loader_columns
            ]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.hparams.task_name)
        self.hparams.tokenizer_cls.from_pretrained(self.hparams.bert_model)

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=8,
            shuffle=True,
        )

    @property
    def test_batch_size(self):
        return max(self.hparams.batch_size // 4, 2)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["validation"],
                batch_size=self.test_batch_size,
                pin_memory=True,
                num_workers=8,
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(
                    self.dataset[x],
                    batch_size=self.test_batch_size,
                    pin_memory=True,
                    num_workers=8,
                )
                for x in self.eval_splits
            ]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["test"],
                batch_size=self.test_batch_size,
                pin_memory=True,
                num_workers=8,
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(
                    self.dataset[x],
                    batch_size=self.test_batch_size,
                    pin_memory=True,
                    num_workers=8,
                )
                for x in self.eval_splits
            ]

    def convert_to_features(self, batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(
                zip(
                    batch[self.text_fields[0]],
                    batch[self.text_fields[1]],
                )
            )
        else:
            texts_or_text_pairs = batch[self.text_fields[0]]

        features = self.tokenizer(
            texts_or_text_pairs,
            max_length=self.hparams.max_input_length,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=TruncationStrategy.LONGEST_FIRST,
            return_tensors=TensorType.NUMPY,
        )

        features["labels"] = batch["label"]

        return features
