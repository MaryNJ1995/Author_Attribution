import torch
from torch import nn
import torch.nn.functional as function

__author__ = "Maryam Najafi🥰"
__organization__ = "Religious ChatBot"
__license__ = "Public Domain"
__version__ = "1.1.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__status__ = "Production"
__date__ = "07/27/2021"
"""
In this class we implement Attention model for"""


class Attention2(nn.Module):
    """
    In this class we implement Attention model for text classification
    """

    def __init__(self, rnn_size):
        super(Attention2, self).__init__()

    def forward(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        concat_state = torch.cat([s for s in final_state], 1)
        concat_state = concat_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, concat_state)
        weights = function.softmax(weights.squeeze(2)).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

# powered by: https://github.com/Renovamen/Text-Classification/tree/master/models/AttBiLSTM
# todo https://github.com/ddhruvkr/Deep-Learning-Text-Classification/blob/master/pytorch_models/Liu_InnerAttention.py
# todo https://github.com/ddhruvkr/Deep-Learning-Text-Classification/blob/master/pytorch_models/ConvRNN.py
# todo https://github.com/ddhruvkr/Deep-Learning-Text-Classification/blob/master/pytorch_models/BIDAF.py
