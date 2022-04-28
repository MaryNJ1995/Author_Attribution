import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AdamW, BertForNextSentencePrediction, get_linear_schedule_with_warmup
from src.configuration.config import BaseConfig
from src.model.attention_model import Attention

__author__ = "Maryam Najafi🥰"
__organization__ = "Religious ChatBot"
__license__ = "Public Domain"
__version__ = "1.1.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__status__ = "Production"
__date__ = "07/27/2021"


class Classifier(pl.LightningModule):
    def __init__(self, arg, n_classes, steps_per_epoch):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.f1_score = torchmetrics.F1(average="weighted", num_classes=n_classes)
        self.category_f1_score = torchmetrics.F1(average="none", num_classes=n_classes)
        self.lr = arg.lr
        self.n_epochs = arg.n_epochs
        self.steps_per_epoch = steps_per_epoch  # BCEWithLogitsLoss
        self.criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss()
        self.bert = BertModel.from_pretrained(arg.bert_model_path, return_dict=True)
        self.lstm1 = torch.nn.LSTM(input_size=768,
                                   hidden_size=128,
                                   num_layers=1,
                                   bidirectional=True)
        self.lstm2 = torch.nn.LSTM(input_size=2 * 128,
                                   hidden_size=128,
                                   num_layers=1,
                                   bidirectional=True)
        self.fully_connected_layers = torch.nn.Sequential(
            torch.nn.Linear(30 * ((4 * 128) + 768), 2000),
            torch.nn.Linear(2000, n_classes)
        )

        self.attention = Attention(4 * 128 + 768)

        self.dropout = torch.nn.Dropout(0.2)
        self.save_hyperparameters()

    def forward(self, input_ids, attn_mask, token_type_ids):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attn_mask,
                                token_type_ids=token_type_ids)
        # bert_output.last_hidden_state.size() = [batch_size, sen_len, 768]

        embedded = bert_output.last_hidden_state.permute(1, 0, 2)
        # embedded.size() =  [sen_len, batch_size, 768]

        output1, (hidden, cell) = self.lstm1(embedded)
        output2, (hidden, cell) = self.lstm2(output1, (hidden, cell))
        # output.size() = [sent_len, batch_size, hid_dim * num_directions]
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # cell.size() = [num_layers * num_directions, batch_size, hid_dim]

        outputs_cat = torch.cat((output1, output2, embedded), dim=2)
        # outputs_cat.size()= (sent_len, batch_size, (2*hid_dim * num_directions+emb_dim))==>torch.Size([150, 64, 1324])

        attention_score = self.attention(outputs_cat)
        # this score is the importance of each word in 30 different way
        # attention_score.size() = (batch_size, num_head_attention, sent_len)===>([64, 30, 150])
        outputs_cat = outputs_cat.permute(1, 0, 2)
        hidden_matrix = torch.bmm(attention_score, outputs_cat)

        # hidden_matrix is each word's importance * each word's feature
        # hidden_matrix.size() = (batch_size, num_head_attention,  hid_dim * num_directions])===>([64, 30, 512])

        # concatenate the hidden_matrix:
        final_output = self.fully_connected_layers(
            hidden_matrix.view(-1, hidden_matrix.size()[1] *  # num_head_attention
                               hidden_matrix.size()[2]))  # hid_dim * num_directions
        # hidden_matrix.view.size(64, 512*30)
        # final_output.size() = (batch_size, output_size)===>([64, 2])

        return final_output

    def training_step(self, batch, batch_idx):
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
        optimizer = AdamW(self.parameters(), lr=self.lr)
        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps
        # scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer]  # , [scheduler]


if __name__ == '__main__':
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()
    MODEL = Classifier(max_length=CONFIG.max_length, n_classes=2,
                       steps_per_epoch=10,
                       n_epochs=CONFIG.n_epochs, lr=CONFIG.lr,
                       bert_model_path=CONFIG.bert_model)
    x = torch.rand((64, 150))
    y = torch.rand((64, 150))
    z = torch.rand((64, 150))

    MODEL.forward(x.long(), y.long(), z.long())
