# -*- coding: utf-8 -*-
# @Time: 2021/4/11 11:01
# @Author: Zhou Quan
# @Software: PyCharm

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class Gait_Dataset(Dataset):
    """Personal-specific disentangle Model"""
    def __init__(self, csv_file, sensor_channels, seq_len, predict_data_channels, motion_label=None, root_dir=None, transform=None):
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.sensor_channels = sensor_channels
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.predict_data_channels = predict_data_channels
        self.motion_label = motion_label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample_data_len = self.sensor_channels * self.seq_len
        sample_data = self.data.iloc[index, 0: sample_data_len].values.astype('float')
        predict_data = self.data.iloc[index, sample_data_len: -2].values.astype('float')
        motion_label = self.data.iloc[index, -2]
        person_id = self.data.iloc[index, -1]

        """Reshape the Data"""
        sample_data = np.reshape(sample_data, [self.seq_len, self.sensor_channels])

        return sample_data, motion_label, person_id
