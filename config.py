# -*- coding: utf-8 -*-
# @Time: 2021/4/11 10:59
# @Author: Zhou Quan
# @Software: PyCharm

import torch.nn as nn


class Model_Config:
    def __init__(self):
        self.learning_rate = 0.001
        self.epochs = 20
        self.dropout = 0.5
        self.log_interval = 100
        self.batch_size = 64
        self.ksize = 3
        self.random_seed = 1234
        self.clip = -1
        self.reg_criterion = nn.MSELoss()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.id_criterion = nn.CrossEntropyLoss()
        # Alpha and beta should be changed in different data or tasks
        self.alpha = 0.4
        self.beta = 0.9

