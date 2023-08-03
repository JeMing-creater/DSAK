import torch
from torch import nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F

import random


class AlexNet(nn.Module):
    def __init__(self,in_channels=3,num_classes=10):
        super(AlexNet, self).__init__()

        self.Conv = nn.Sequential(
            # IN : 3*32*32
            nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=5, stride=2, padding=2),
            # 论文中kernel_size = 11,stride = 4,padding = 2
            nn.ReLU(),
            # IN : 96*16*16
            nn.MaxPool2d(kernel_size=2, stride=2),  # 论文中为kernel_size = 3,stride = 2
            # IN : 96*8*8
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # IN :256*8*8
            nn.MaxPool2d(kernel_size=2, stride=2),  # 论文中为kernel_size = 3,stride = 2
            # IN : 256*4*4
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.MaxPool2d(kernel_size=2, stride=2),  # 论文中为kernel_size = 3,stride = 2
            # OUT : 384*2*2
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=384 * 2 * 2, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self,x):
        x = self.Conv(x)
        x = x.view(-1, 384 * 2 * 2)
        x = self.linear(x)
        return x


    # train model

# class AlexNet(nn.Module):
#     def __init__(self,in_channels=3,out_features=10):
#         super(AlexNet, self).__init__()
#         self.Conv = nn.Sequential(
#             # IN : 3*32*32
#             nn.Conv2d(in_channels=in_channels,out_channels=96,kernel_size=5,stride=2,padding=2),      # 论文中kernel_size = 11,stride = 4,padding = 2
#             nn.ReLU(),
#             # IN : 96*16*16
#             nn.MaxPool2d(kernel_size=2,stride=2),              # 论文中为kernel_size = 3,stride = 2
#             # IN : 96*8*8
#             nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             # IN :256*8*8
#             nn.MaxPool2d(kernel_size=2,stride=2),              # 论文中为kernel_size = 3,stride = 2
#             # IN : 256*4*4
#             nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             # IN : 384*4*4
#             nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             # IN : 384*4*4
#             nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             # IN : 384*4*4
#             nn.MaxPool2d(kernel_size=2, stride=2),              # 论文中为kernel_size = 3,stride = 2
#             # OUT : 384*2*2
#         )
#         self.linear = nn.Sequential(
#             nn.Linear(in_features=384 * 2 * 2, out_features=4096),
#             nn.ReLU(),
#             nn.Linear(in_features=4096, out_features=4096),
#             nn.ReLU(),
#             nn.Linear(in_features=4096, out_features=out_features),
#         )
#     def forward(self,x):
#             x = self.Conv(x)
#             x = x.view(-1, 384 * 2 * 2)
#             x = self.linear(x)
#             return x
