import torch
import torch.nn.functional as function
from torch import nn

__author__ = "Maryam Najafi🥰"
__organization__ = "Author Attribution"
__license__ = "Public Domain"
__version__ = "1.0.0"
__email__ = "Maryaminj1995@gmail.com"
__status__ = "Production"
__date__ = "07/27/2021"
"""
In this class we implement Attention model """


class Attention(nn.Module):
    """
    In this class we implement Attention model for text classification
    """

    def __init__(self, rnn_size):
        super(Attention, self).__init__()
        self.att_fully_connected_layers1 = nn.Linear(rnn_size, 350)
        self.att_fully_connected_layers2 = nn.Linear(350, 30)

    '''
    input param: 
        lstm_output: output of bi-LSTM (batch_size, sent_len, hid_dim * num_directions)   
    return: 
        attention weights matrix, 
        attention_output(batch_size, num_directions * hidden_size): attention output vector for the batch
        with this informations:
        r=30 and da=350 and penalize_coeff = 1
    '''

    def forward(self, lstm_output):
        # work_flow: lstm_output>fc1>tanh>fc2>softmax
        # usage of softmax is to calculate the distribution probability through one sentence :)
        lstm_output = lstm_output.permute(1, 0, 2)
        hidden_matrix = self.att_fully_connected_layers2(torch.tanh(self.att_fully_connected_layers1(lstm_output)))
        # hidden_matrix.size() :(batch_size, sent_len, num_head_attention)===> torch.Size([64, 150, 30])
        # for each of this 150 word, we have 30 feature from 30 attention's head.
        # permute? because the softmax will apply in 3rd dimension and we want apply it on sent_len so:
        hidden_matrix = hidden_matrix.permute(0, 2, 1)
        # hidden_matrix.size() :(batch_size, num_head_attention, sent_len)===>torch.Size([64, 30, 150])
        hidden_matrix = function.softmax(hidden_matrix, dim=2)
        # hidden_matrix.size() :(batch_size, num_head_attention, sent_len)===>torch.Size([64, 30, 150])
        # print(hidden_matrix.size())

        return hidden_matrix
# powered by: https://github.com/Renovamen/Text-Classification/tree/master/models/AttBiLSTM
# todo https://github.com/ddhruvkr/Deep-Learning-Text-Classification/blob/master/pytorch_models/Liu_InnerAttention.py
# todo https://github.com/ddhruvkr/Deep-Learning-Text-Classification/blob/master/pytorch_models/ConvRNN.py
# todo https://github.com/ddhruvkr/Deep-Learning-Text-Classification/blob/master/pytorch_models/BIDAF.py
