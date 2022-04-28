"""
This module is to implement a new structure for matching Semantically simillar text
"""
import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as function
from configuration.config import BaseConfig
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AdamW

__author__ = "Maryam Najafi🥰"
__organization__ = "Religious ChatBot"
__license__ = "Public Domain"
__version__ = "1.1.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__status__ = "Production"
__date__ = "07/27/2021"


class Classifier(pl.LightningModule):
    """
    a class for implement deep+LM architeture for matching semantically simillar texts
    """

    def __init__(self, arg, n_classes, steps_per_epoch):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.f1_score = torchmetrics.F1(average="weighted", num_classes=n_classes)
        self.category_f1_score = torchmetrics.F1(average="none", num_classes=n_classes)
        self.lr = arg.lr
        self.num_layers = arg.num_layers
        self.hidden_size = arg.hidden_size
        self.input_size = arg.input_size
        self.n_epochs = arg.n_epochs
        self.out_features = 256
        self.steps_per_epoch = steps_per_epoch  # BCEWithLogitsLoss
        self.criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss()
        self.bert = BertModel.from_pretrained(arg.bert_model_path, return_dict=True)
        self.lstm = torch.nn.LSTM(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=True)
        self.fully_connected_layers_cat = \
            torch.nn.Linear(in_features=self.input_siz + (2 * self.hidden_size),
                            out_features=self.out_features)
        self.fully_connected_layers_last = \
            torch.nn.Linear(in_features=self.out_features,
                            out_features=n_classes)

        self.dropout = torch.nn.Dropout(arg.dropout)
        self.save_hyperparameters()
        self.tanh = torch.nn.Tanh()

    def forward(self, input_ids, attn_mask, token_type_ids):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attn_mask,
                                token_type_ids=token_type_ids)
        # bert_output.last_hidden_state.size() = [batch_size, sen_len, 768]

        embedded = bert_output.last_hidden_state.permute(1, 0, 2)
        # embedded.size() =  [sen_len, batch_size, 768]

        output, (hidden, cell) = self.lstm(embedded)
        # output.size() = [sent_len, batch_size, hid_dim * num_directions]
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # cell.size() = [num_layers * num_directions, batch_size, hid_dim]

        output = torch.cat([output, embedded], dim=2)
        output = output.permute(1, 0, 2)
        # output.size() = [batch_size, sent_len, embedding_dim+2*hid_dim]===>(64,150,1024)

        output = self.tanh(self.fully_connected_layers_cat(output))
        # output.size() = [batch_size, sent_len, out_units]===>(64,150,256)

        # prepare data for max pooling:
        output = output.permute(0, 2, 1)
        # output.size() = [batch_size, out_units, sent_len]===>(64,256,150)

        output = function.max_pool1d(output, output.shape[2]).squeeze(2)
        # output.size() = [batch_size, out_units]===>(64,256)

        output = self.fully_connected_layers_last(output)
        # output.size() = [batch_size,output_size]===>(64,2)
        # print(output.size())

        return output

    def training_step(self, batch, batch_idx):
        """
        configure a test step with caculatiing loss and accuracy and logging the results
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["label"].flatten()
        outputs = self.forward(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(outputs, labels)
        self.log("train_acc", self.accuracy(torch.softmax(outputs, dim=1), labels),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1", self.f1_score(torch.softmax(outputs, dim=1), labels),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_first_class", self.category_f1_score(
            torch.softmax(outputs, dim=1), labels)[0],
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_second_class", self.category_f1_score(
            torch.softmax(outputs, dim=1), labels)[1],
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        """
        configure a test step with caculatiing loss and accuracy and logging the results
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["label"].flatten()
        outputs = self.forward(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(outputs, labels)
        self.log("val_acc", self.accuracy(torch.softmax(outputs, dim=1), labels),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", self.f1_score(torch.softmax(outputs, dim=1), labels),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_first_class", self.category_f1_score(
            torch.softmax(outputs, dim=1), labels)[0],
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_second_class", self.category_f1_score(
            torch.softmax(outputs, dim=1), labels)[1],
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        configure a test step with caculatiing loss and accuracy and logging the results
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["label"].flatten()
        outputs = self.forward(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(outputs, labels)
        self.log("test_acc", self.accuracy(torch.softmax(outputs, dim=1), labels),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1", self.f1_score(torch.softmax(outputs, dim=1), labels),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1_first_class", self.category_f1_score(
            torch.softmax(outputs, dim=1), labels)[0],
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1_second_class", self.category_f1_score(
            torch.softmax(outputs, dim=1), labels)[1],
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """
        configure a suitable optimizer base on warm up on step per epoch
        """
        optimizer = AdamW(self.parameters(), lr=self.lr)
        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps
        # scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer]  # , [scheduler]


if __name__ == '__main__':
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()
    MODEL = Classifier(CONFIG, n_classes=2,
                       steps_per_epoch=10)

    X = torch.rand((64, 150))
    Y = torch.rand((64, 150))
    Z = torch.rand((64, 150))

    MODEL.forward(X.long(), Y.long(), Z.long())
