from argparse import ArgumentParser

import torch.nn.functional as F
from smart_compress.models.base import BaseModule
from transformers import BertForSequenceClassification, BertTokenizer


class BertModule(BaseModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(
            parents=[BaseModule.add_argparse_args(parent_parser)], add_help=False
        )
        parser.add_argument("--num_labels", default=2, type=int)
        parser.add_argument("--bert_model", default="bert-base-uncased", type=str)
        parser.add_argument("--dropout_probability", default=0.3, type=float)
        parser.add_argument("--no_freeze", action="store_false", dest="freeze")
        return parser

    def __init__(self, *args, tokenizer_cls=BertTokenizer, **kwargs):
        super(BertModule, self).__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.model = BertForSequenceClassification.from_pretrained(
            self.hparams.bert_model, num_labels=self.hparams.num_labels
        )

        if self.hparams.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def loss_function(self, outputs, ground_truth):
        return F.cross_entropy(outputs, ground_truth)

    def calculate_loss(self, batch):
        inputs, labels = batch

        outputs = self(inputs, labels)
        loss = outputs.loss
        if self.hparams.compress_loss:
            loss = self.compression(loss, self.hparams)

        return labels, loss, outputs

    def forward(self, x, labels):
        return self.model(**x, labels=labels)
