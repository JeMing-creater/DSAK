
import torch

import torch.nn as nn


# 以模型形式交付交叉熵损失函数
class CrossEntropy_L2(nn.Module):
    '''
    交叉熵损失函数
    '''
    def __init__(self, model, m, l2_ratio):
        super(CrossEntropy_L2, self).__init__()
        self.model = model
        self.m = m
        self.w = 0.0
        self.l2_ratio = l2_ratio

    def forward(self, y_pred, y_test):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_pred, y_test)
        for name in self.model.state_dict():
            if name.find('weight') != -1:
                self.w += torch.sum(torch.square(self.model.state_dict()[name]))
        loss = torch.add(torch.mean(loss), self.l2_ratio * self.w / self.m / 2)

        return loss
