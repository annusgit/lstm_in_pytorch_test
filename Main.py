

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

class LSTM(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(hidden_size, hidden_size),
            nn.LSTM(hidden_size, hidden_size),
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax()
        )

    def forward(self, x):

        pass

    def train_model(self):

        pass

    def test_model(self):
        pass



