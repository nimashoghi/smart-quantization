from argparse import ArgumentParser

import torch.nn as nn
import torch.nn.functional as F
from smart_compress.models.base import BaseModule
from transformers import BertModel, BertTokenizer


class BertModule(BaseModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[BaseModule.add_model_specific_args(parent_parser)], add_help=False
        )
        parser.add_argument("--input_length", default=512, type=int)
        parser.add_argument("--output_size", default=1, type=int)
        parser.add_argument("--bert_model", default="bert-base-uncased", type=str)
        parser.add_argument("--dropout_probability", default=0.3, type=float)
        return parser

    def __init__(self, *args, tokenizer_cls=BertTokenizer, **kwargs):
        super(BertModule, self).__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.model = BertModel.from_pretrained(self.hparams.bert_model)
        self.drop = nn.Dropout(p=self.hparams.dropout_probability)
        self.out = nn.Linear(self.model.config.hidden_size, self.hparams.output_size)

    def loss_function(self, outputs, ground_truth):
        return F.cross_entropy(outputs, ground_truth)

    def forward(self, x):
        output = self.model(input_ids=x["input_ids"], attention_mask=x["input_mask"])
        output = self.drop(output.pooler_output)
        return self.out(output)
