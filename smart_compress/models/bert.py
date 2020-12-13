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
        parser.add_argument("--output_size", default=2, type=int)
        parser.add_argument("--bert_model", default="bert-base-uncased", type=str)
        parser.add_argument("--dropout_probability", default=0.3, type=float)
        parser.add_argument(
            "--no_freeze_bert", action="store_false", dest="freeze_bert"
        )
        return parser

    def __init__(self, *args, tokenizer_cls=BertTokenizer, **kwargs):
        super(BertModule, self).__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.model = BertModel.from_pretrained(self.hparams.bert_model)
        self.fc1 = nn.Linear(
            self.model.config.hidden_size, self.model.config.hidden_size
        )
        self.fc2 = nn.Linear(self.model.config.hidden_size, self.hparams.output_size)

        if self.hparams.freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False

    def loss_function(self, outputs, ground_truth):
        return F.cross_entropy(outputs, ground_truth)

    def forward(self, x):
        x = F.relu(self.model(**x).pooler_output)
        x = F.dropout(x, p=self.hparams.dropout_probability)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
