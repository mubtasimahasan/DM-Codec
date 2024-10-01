# Reference: https://arxiv.org/pdf/2210.13438

"""LSTM layers module."""

from torch import nn


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True, bidirectional: bool=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers, bidirectional=bidirectional)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.bidirectional:
            x = x.repeat(1, 1, 2)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y
