# -*- coding: utf-8 -*-
# @Software: PyCharm

import torch
import torch.nn as nn


class Simple_LSTM_with_disentangle_Model(nn.Module):
    """No attention, No PR, just simple make prediction and recognition"""
    def __init__(self, sensor_channels, hidden_channels, num_classes=5, num_layers=2):
        super(Simple_LSTM_with_disentangle_Model, self).__init__()
        self.hidden_size = hidden_channels
        self.num_layers = num_layers
        self.lstm = nn.LSTM(sensor_channels, hidden_channels, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_channels, hidden_channels)
        self.r1 = nn.Linear(hidden_channels, 20)
        self.r2 = nn.Linear(20, num_classes)

        """Parallel Module"""
        self.parallel_module = nn.Sequential(nn.Conv2d(1, 12, kernel_size=3, stride=1),
                                             nn.ReLU(inplace=True),
                                             nn.AvgPool2d(2, 2))
        self.fc1 = nn.Linear(17 * 4 * 12, 40)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 7)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        # Forward propagate LSTM
        out, (final_hidden_state, final_cell_state) = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        encoded_vector = self.fc(out[:, -1, :])
        # print(encoded_vector.shape)
        recognition_out = self.r1(encoded_vector)
        recognition_out = self.r2(recognition_out)

        """Parallel Model"""
        parallel_input = x.unsqueeze(1)
        parallel_out = self.parallel_module(parallel_input)
        parallel_out = parallel_out.view(-1, 17 * 4 * 12)
        # print(parallel_out)
        parallel_out = self.fc1(parallel_out)
        parallel_out = self.fc2(parallel_out)
        # print(parallel_out.shape)
        """Recognize the Person ID"""
        person_id = self.fc3(parallel_out)

        """Inner Product"""
        sep_loss = torch.mul(final_hidden_state, parallel_out)

        return recognition_out, person_id, sep_loss
