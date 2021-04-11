# -*- coding: utf-8 -*-
# @Time: 2021/4/11 10:56
# @Author: Zhou Quan
# @Software: PyCharm

import torch
import torch.nn as nn
from Personal_specific_gait_recognition.models import Simple_LSTM_with_disentangle_Model
from Personal_specific_gait_recognition.config import Model_Config
from torch.utils.data import DataLoader
from Personal_specific_gait_recognition.utils import Gait_Dataset
from tensorboardX import SummaryWriter

writer = SummaryWriter('personal_specific_log_1')

"""Loading Data"""
train_file = r'F:\Datasets\Huake_gait_data\try_build_new_dataset\new_train_data.csv'
val_file = r'F:\Datasets\Huake_gait_data\try_build_new_dataset\new_test_data.csv'

"""Model Selection"""
model = Simple_LSTM_with_disentangle_Model(sensor_channels=36, hidden_channels=40, num_classes=5)

"""Hyper-parameters"""
config = Model_Config()
lr = config.learning_rate
log_interval = config.log_interval
epochs = config.epochs
batch_size = config.batch_size
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
clip = config.clip
reg_criterion = config.reg_criterion
cls_criterion = config.cls_criterion
id_criterion = config.id_criterion
alpha = config.alpha
beta = config.beta

print(config)

train_loader = DataLoader(Gait_Dataset(train_file, 36, 10, 36), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Gait_Dataset(val_file, 36, 10, 36), batch_size=batch_size)
print(train_loader)
print(val_loader)


# Define Model
model_file = "Simple_LSTM_Disentangle_1"

# Use CUDA
if torch.cuda.is_available():
    model.cuda()

print(model)


# Training Model
def train(epoch):
    total = 0
    correct = 0
    model.train()
    for batch_idx, (sample_data, motion_label, person_id) in enumerate(train_loader):
        sample_data = sample_data.type(torch.FloatTensor)
        motion_label = motion_label.type(torch.LongTensor)
        person_id = person_id.type(torch.LongTensor)

        sample_data, motion_label, person_id = sample_data.cuda(), motion_label.cuda(), person_id.cuda()

        optimizer.zero_grad()
        motion, person_rec, sep_loss = model(sample_data)

        cls_loss = cls_criterion(motion, motion_label)

        id_loss = id_criterion(person_rec, person_id)
        total_sep_loss = torch.sum(abs(sep_loss))

        total_loss = cls_loss + alpha * id_loss + beta * total_sep_loss

        total_loss.backward()

        _, motion = torch.max(motion.data, 1)
        total += motion_label.size(0)
        correct += (motion == motion_label).sum().item()

        writer.add_scalar('Training Accuracy', 100 * correct / total, epoch)

        if clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        optimizer.step()

        if batch_idx > 0 and batch_idx % log_interval == 0:
            print('Train Epoch: {} \t total Loss:{:.6f}'.format(epoch, cls_loss))


def val():
    model.eval().cuda()
    correct = 0
    total = 0
    for batch_idx, (sample_data, motion_label, person_id) in enumerate(val_loader):

        sample_data = sample_data.type(torch.FloatTensor)
        motion_label = motion_label.type(torch.LongTensor)
        person_id = person_id.type(torch.LongTensor)

        sample_data, motion_label, person_id = sample_data.cuda(), motion_label.cuda(), person_id.cuda()

        optimizer.zero_grad()

        motion, person_rec, sep_loss = model(sample_data)

        _, motion = torch.max(motion.data, 1)
        total += motion_label.size(0)
        correct += (motion == motion_label).sum().item()

        writer.add_scalar('Test Accuracy', 100 * correct/total, epoch)

        if clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)

    print("Validation Accuracy: {}".format(100 * correct / total))

    return 100 * correct / total


if __name__ == '__main__':
    min_acc = 10
    for epoch in range(1, epochs + 1):
        train(epoch)
        acc = val()
        if acc > min_acc:
            min_acc = acc
            print("save model")
            torch.save(model.state_dict(), 'personal_specific_log_1.pth')
