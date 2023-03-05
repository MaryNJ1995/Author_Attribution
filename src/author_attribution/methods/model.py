# !/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ
# pylint: disable-msg=import-error
# pylint: disable=too-many-instance-attributes
"""
DCBiLstm_model.py is written for DC_BiLstm model
"""

import torch
from torch import nn
from author_attribution.methods.attention_model import Attention

__author__ = "Maryam Najafi"
__organization__ = "AuthorShip Attribution"
__license__ = "Public Domain"
__version__ = "1.0.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__status__ = "Production"
__date__ = "06/09/2021"


class DCBiLstmAttentions(nn.Module):
    """
    In this class we implement DC_BiLstm model
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=kwargs["vocab_size"],
                                       embedding_dim=kwargs["embedding_dim"],
                                       padding_idx=kwargs["pad_idx"])

        self.lstm_1 = nn.LSTM(input_size=kwargs["embedding_dim"],
                              hidden_size=kwargs["lstm_units"],
                              num_layers=1,
                              bidirectional=kwargs["bidirectional"])

        self.lstm_2 = nn.LSTM(input_size=
                              2 * kwargs["lstm_units"] + kwargs["embedding_dim"],
                              hidden_size=kwargs["lstm_units"],
                              num_layers=1,
                              bidirectional=kwargs["bidirectional"])

        self.lstm_3 = nn.LSTM(input_size=
                              4 * kwargs["lstm_units"] + kwargs["embedding_dim"],
                              hidden_size=kwargs["lstm_units"],
                              num_layers=1,
                              bidirectional=kwargs["bidirectional"])

        self.attention1 = Attention(
            (3 * (2 * kwargs["lstm_units"])) + kwargs["embedding_dim"])

        self.dropout = nn.Dropout(kwargs["dropout"])

        self.fully_connected_layers = \
            nn.Linear(30 * ((3 * (2 * kwargs["lstm_units"])) +
                            kwargs["embedding_dim"]), kwargs["output_size"])

        self.tanh = nn.Tanh()

    def forward(self, input_batch):
        """
        to make the best path for parameters
        """
        # input_batch.size() = [batch_size, sent_len]
        # input_pos.size() = [batch_size, sent_len]

        word_embedded = self.embeddings(input_batch)
        # embedded.size() = [batch_size, sent_len, embedding_dim]===>torch.Size([64, 150, 300])
        # embedded.size() = [batch_size, sent_len, pos_embedding_dim]

        # embedded.size() = [batch_size, sent_len, embedding_dim+pos_embedding_dim]

        embedded = word_embedded.permute(1, 0, 2)

        embedded = self.dropout(embedded)
        # embedded.size() = [sent_len, batch_size, embedding_dim]
        # ===>torch.Size([150, 64, 300])

        output_1, (hidden, cell) = self.lstm_1(embedded)
        # output_1.size() = [sent_len, batch_size, hid_dim*num_directions]
        # ===>torch.Size([150, 64, 2*256])
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # cell.size() = [num_layers * num_directions, batch_size, hid_dim]

        lstm_2_input = torch.cat((output_1, embedded), dim=2)
        # lstm_2_input.size() = [sent_len, batch_size, hid_dim*num_directions+embedding_dim]
        # torch.Size([150, 64, (2*256+300)])

        output_2, (hidden, cell) = self.lstm_2(lstm_2_input, (hidden, cell))
        # output_2.size() = [sent_len, batch_size, hid_dim*num_directions]
        # ===>torch.Size([150, 64, 2*256])
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
        # all_features.size===>
        # [sent_len, batch_size, (3*hid_dim * num_directions)+ pos_embedding_dim + embedding_dim]

        attention_score = self.attention1(all_features)
        # this score is the importance of each word in 30 different way
        # attention_score.size() = [batch_size, num_head_attention, sent_len]===>([64, 30, 150])

        all_features = all_features.permute(1, 0, 2)
        # all_features.size() ===>
        # [batch_size, sent_len, (3*hid_dim * num_directions)+ pos_embedding_dim + embedding_dim]

        hidden_matrix = torch.bmm(attention_score, all_features)
        # hidden_matrix is each word's importance * each word's feature
        # hidden_matrix.size===>
        # (batch_size,num_head_attention,(3*hid_dim*num_directions)+pos_embedding_dim+embedding_dim)

        # concatenate the hidden_matrix:
        final_output = (self.fully_connected_layers(
            hidden_matrix.view(-1, hidden_matrix.size()[1] *  # num_head_attention
                               hidden_matrix.size()[2])))
        # hidden_matrix.size()[2].size()=(3*hid_dim*num_directions)+pos_embedding_dim+embedding_dim
        return final_output


if __name__ == "__main__":
    MODEL = DCBiLstmAttentions(vocab_size=2000,
                               embedding_dim=300,
                               lstm_units=256, output_size=2,
                               lstm_layers=2, bidirectional=True,
                               start_dropout=0.5, middle_dropout=0.2,
                               pad_idx=1, final_dropout=0.2, dropout=0.2)
    X = torch.rand((64, 150))
    Y = torch.rand((64, 150))

    MODEL.forward(X.long())
