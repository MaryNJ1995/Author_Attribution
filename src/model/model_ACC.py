import pytorch_lightning as pl
import torch
import torch.nn.functional as function
import torchmetrics
from configuration.config import BaseConfig
from model.attention_model import Attention
from transformers import BertModel, AdamW

__author__ = "Maryam Najafi🥰"
__organization__ = "Author Attribution"
__license__ = "Public Domain"
__version__ = "1.0.0"
__email__ = "Maryaminj1995@gmail.com"
__status__ = "Production"
__date__ = "07/27/2021"


class Classifier(pl.LightningModule):
    def __init__(self, arg, steps_per_epoch, n_classes):
        super().__init__()

        self.lstm_input_shape = self.lstm_input(filter_sizes=arg.filter_sizes, max_len=arg.max_len)

        self.accuracy = torchmetrics.Accuracy()
        self.f1_score = torchmetrics.F1(average="weighted", num_classes=n_classes)
        self.category_f1_score = torchmetrics.F1(average="none", num_classes=n_classes)
        self.lr = arg.lr
        self.n_filters = arg.n_filters
        self.hidden_size = arg.hidden_size
        self.embedding_size = arg.embedding_size
        self.n_epochs = arg.n_epochs
        self.filter_sizes = arg.filter_sizes
        self.num_layers = arg.num_layers
        self.steps_per_epoch = steps_per_epoch  # BCEWithLogitsLoss
        self.criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss()
        self.bert = BertModel.from_pretrained(arg.bert_model_path, return_dict=True)
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1,
                            out_channels=self.n_filters,
                            kernel_size=(fs, self.embedding_size))
            for fs in self.filter_sizes])
        self.lstm = torch.nn.LSTM(self.lstm_input_shape,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=True)

        self.attention = Attention(rnn_size=2 * self.hidden_size)
        # We will use da = 350, r = 30 & penalization_coeff = 1
        # as per given in the self-attention original ICLR paper
        self.W_s1 = torch.nn.Linear(2 * self.hidden_size, 350)
        self.W_s2 = torch.nn.Linear(350, 30)

        self.fully_connected_layers = torch.nn.Sequential(
            torch.nn.Linear(30 * 2 * self.hidden_size, 2000),
            torch.nn.Linear(2000, n_classes)
        )

        self.dropout = torch.nn.Dropout(0.2)
        self.save_hyperparameters()

    @staticmethod
    def lstm_input(filter_sizes, max_len):
        lstm_input_shape = 0
        for item in filter_sizes:
            lstm_input_shape += (max_len - item + 1)
        return lstm_input_shape

    def forward(self, input_ids, attn_mask, token_type_ids):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attn_mask,
                                token_type_ids=token_type_ids)
        # bert_output.last_hidden_state.size() = [batch_size, sen_len, emb_dim]

        embedded = bert_output.last_hidden_state.unsqueeze(1)
        # embedded.size() =  [sen_len, batch_size, emb_dim]

        conved = [function.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        conved = torch.cat(conved, dim=2)
        conved = conved.permute(1, 0, 2)
        # conved = [n_filter, batch size, sum[sent len - filter_sizes[n] + 1]]

        lstm_output, (_, _) = self.lstm(conved)
        # output.size() = (batch_size, n_filter, 2*lstm_units)

        attn_weight_matrix = self.attention(lstm_output)
        # attn_weight_matrix.size() = (batch_size, r, n_filter)
        # output.size() = (batch_size, n_filter, 2*lstm_units)
        lstm_output = lstm_output.permute(1, 0, 2)

        hidden_matrix = torch.bmm(attn_weight_matrix, lstm_output)
        # hidden_matrix.size() = (batch_size, r, 2*lstm_units)

        final_output = self.fully_connected_layers(hidden_matrix.view(-1,
                                                                      hidden_matrix.size()[1] * hidden_matrix.size()[
                                                                          2]))

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

    MODEL = Classifier(CONFIG, steps_per_epoch=10, n_classes=2)
    x = torch.rand((64, 150))
    y = torch.rand((64, 150))
    z = torch.rand((64, 150))

    MODEL.forward(x.long(), y.long(), z.long())
