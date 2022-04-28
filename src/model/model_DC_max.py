import torch
import torchmetrics
import pytorch_lightning as pl
from transformers import BertModel, AdamW
from configuration.config import BaseConfig
from model.attention_model import Attention
import torch.nn.functional as function

__author__ = "Maryam Najafi🥰"
__organization__ = "Religious ChatBot"
__license__ = "Public Domain"
__version__ = "1.1.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__date__ = "07/27/2021"


class Classifier(pl.LightningModule):
    def __init__(self, arg, n_classes, steps_per_epoch):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.f1_score = torchmetrics.F1(average="weighted", num_classes=n_classes)
        self.category_f1_score = torchmetrics.F1(average="none", num_classes=n_classes)
        self.lr = arg.lr
        self.hidden_size = arg.hidden_size
        self.input_size = arg.input_size
        self.n_epochs = arg.n_epochs
        self.num_layers = arg.num_layers
        self.steps_per_epoch = steps_per_epoch  # BCEWithLogitsLoss
        self.criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss()
        self.bert = BertModel.from_pretrained(arg.bert_model_path, return_dict=True)
        self.lstm_1 = torch.nn.LSTM(input_size=self.input_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.num_layers,
                                    bidirectional=True)
        self.lstm_2 = torch.nn.LSTM(input_size=(2 * self.hidden_size) + self.input_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.num_layers,
                                    bidirectional=True)
        self.lstm_3 = torch.nn.LSTM(input_size=(4 * self.hidden_size) + self.input_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.num_layers,
                                    bidirectional=True)

        self.attention = Attention(rnn_size=3 * (2 * self.hidden_size) + self.input_size)
        self.fully_connected_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=(2 * ((3 * (2 * self.hidden_size)) + self.input_size)), out_features=750),
            torch.nn.Linear(in_features=750, out_features=n_classes)
        )
        self.batchnorm = torch.nn.BatchNorm1d(num_features=2 * ((3 * (2 * self.hidden_size)) + self.input_size))

        self.dropout = torch.nn.Dropout(0.2)
        self.save_hyperparameters()

    def forward(self, input_ids, attn_mask, token_type_ids):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attn_mask,
                                token_type_ids=token_type_ids)
        # bert_output.last_hidden_state.size() = [batch_size, sen_len, 768]

        embedded = bert_output.last_hidden_state.permute(1, 0, 2)
        # embedded.size() =  [sen_len, batch_size, 768]

        embedded = self.dropout(embedded)
        # embedded.size() = [sent_len, batch_size, embedding_dim]===>torch.Size([150, 64, 768])

        output_1, (hidden, cell) = self.lstm_1(embedded)
        # output_1.size() = [sent_len, batch_size, hid_dim*num_directions]===>torch.Size([150, 64, 2*256])
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # cell.size() = [num_layers * num_directions, batch_size, hid_dim]

        lstm_2_input = torch.cat((output_1, embedded), dim=2)
        # lstm_2_input.size() = [sent_len, batch_size, hid_dim*num_directions+embedding_dim]
        # torch.Size([150, 64, (2*256+300)])

        output_2, (hidden, cell) = self.lstm_2(lstm_2_input, (hidden, cell))
        # output_2.size() = [sent_len, batch_size, hid_dim*num_directions]===>torch.Size([150, 64, 2*256])
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # cell.size() = [num_layers * num_directions, batch_size, hid_dim]

        lstm_3_input = torch.cat((output_1, output_2, embedded), dim=2)
        # lstm_3_input.size() = [sent_len, batch_size, 2*hid_dim*num_directions+embedding_dim]
        # torch.Size([150, 64, (2*2*256+300)])

        output_3, (_, _) = self.lstm_3(lstm_3_input, (hidden, cell))
        # output_3.size() = [sent_len, batch_size, hid_dim * num_directions]
        # _.size() = [num_layers * num_directions, batch_size, hid_dim]
        # _.size() = [num_layers * num_directions, batch_size, hid_dim]

        all_features = torch.cat((output_3, output_2, output_1, embedded), dim=2)
        # all_features.size= [sent_len, batch_size, (3*hid_dim * num_directions)+768]==>torch.Size([150, 64, 1536])

        attention_score = self.attention(all_features)
        # this score is the importance of each word in 30 different way
        # attention_score.size() = [batch_size, num_head_attention, sent_len]===>([64, 30, 150])

        all_features = all_features.permute(1, 0, 2)
        # all_features.size() = [batch_size, sent_len, (3*hid_dim * num_directions)+ pos_embedding_dim + embedding_dim]

        hidden_matrix = torch.bmm(attention_score, all_features)
        hidden_matrix = hidden_matrix.permute(0, 2, 1)
        # hidden_matrix is each word's importance * each word's feature
        # hidden_matrix.size=(batch_size,num_head_attention,((3*hid_dim*num_directions)+768)===>(64,30,1536)
        max_pool_output = function.max_pool1d(hidden_matrix, hidden_matrix.shape[2]).squeeze(2)
        avg_pool_output = function.avg_pool1d(hidden_matrix, hidden_matrix.shape[2]).squeeze(2)

        pooled_cat = torch.cat((max_pool_output, avg_pool_output), dim=1)
        # batchnorm = self.batchnorm(pooled_cat)
        final_output = self.fully_connected_layers(pooled_cat)
        return final_output

    def training_step(self, batch, batch_idx):
        """
        configure a test step with calculating loss and accuracy and logging the results
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
        # self.log("train_f1_first_class", self.category_f1_score(
        #     torch.softmax(outputs, dim=1), labels)[0],
        #          on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train_f1_second_class", self.category_f1_score(
        #     torch.softmax(outputs, dim=1), labels)[1],
        #          on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        """
        configure a test step with calculating loss and accuracy and logging the results
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
        # self.log("val_f1_first_class", self.category_f1_score(
        #     torch.softmax(outputs, dim=1), labels)[0],
        #          on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("val_f1_second_class", self.category_f1_score(
        #     torch.softmax(outputs, dim=1), labels)[1],
        #          on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
        # self.log("test_f1_first_class", self.category_f1_score(
        #     torch.softmax(outputs, dim=1), labels)[8],
        #          on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_f1_second_class", self.category_f1_score(
        #     torch.softmax(outputs, dim=1), labels)[1],
        #          on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
    x = torch.rand((64, 150))
    y = torch.rand((64, 150))
    z = torch.rand((64, 150))

    MODEL.forward(x.long(), y.long(), z.long())
