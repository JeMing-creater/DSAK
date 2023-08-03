import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class Generator(nn.Module):
    def __init__(self, channel=3, size=32, nz=100):
        super(Generator, self).__init__()
        self.ConvTranspose2dStart = nn.ConvTranspose2d(nz, size * 2 * 4, 4, 1, 0, bias=False)
        self.ConvTranspose2d42 = nn.ConvTranspose2d(size * 2 * 4, size * 2 * 2, 4, 2, 1, bias=False)
        self.ConvTranspose2d22 = nn.ConvTranspose2d(size * 2 * 2, size * 2 * 2, 4, 2, 1, bias=False)
        self.ConvTranspose2dEnd = nn.ConvTranspose2d(size * 2 * 2, channel, 4, 2, 1, bias=False)
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.BatchNorm2d4 = nn.BatchNorm2d(size * 2 * 4)
        self.BatchNorm2d2 = nn.BatchNorm2d(size * 2 * 2)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.ConvTranspose2dStart(x)
        x = self.BatchNorm2d4(x)
        x = self.LeakyReLU(x)
        x = self.ConvTranspose2d42(x)
        x = self.BatchNorm2d2(x)
        x = self.LeakyReLU(x)
        x = self.ConvTranspose2d22(x)
        x = self.BatchNorm2d2(x)
        x = self.LeakyReLU(x)
        x = self.ConvTranspose2dEnd(x)
        x = self.Tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, channel=3, size=32):
        super(Discriminator, self).__init__()
        self.Conv2dStart = nn.Conv2d(channel, size, 4, 2, 1, bias=False)
        self.Conv2d22 = nn.Conv2d(size, size * 2, 4, 2, 1, bias=False)
        self.Conv2d24 = nn.Conv2d(size * 2, size * 4, 4, 2, 1, bias=False)
        self.Conv2d44 = nn.Conv2d(size * 4, size * 4, 4, 2, 1, bias=False)
        # self.Conv2d48 = nn.Conv2d(size  * 4, size  *8, 4, 2, 1, bias=False)
        self.Conv2d48 = nn.Conv2d(size * 4, 1, 4, 1, 1, bias=False)
        self.Conv2dEnd = nn.Conv2d(size * 8, 1, 4, 1, 0, bias=False)
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.BatchNorm2d8 = nn.BatchNorm2d(size * 8)
        self.BatchNorm2d4 = nn.BatchNorm2d(size * 4)
        self.BatchNorm2d2 = nn.BatchNorm2d(size * 2)
        self.BatchNorm2d = nn.BatchNorm2d(size)
        self.Flatten = nn.Flatten()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.Conv2dStart(x)
        x = self.BatchNorm2d(x)
        x = self.LeakyReLU(x)
        x = self.Conv2d22(x)
        x = self.BatchNorm2d2(x)
        x = self.LeakyReLU(x)
        x = self.Conv2d24(x)
        x = self.BatchNorm2d4(x)
        x = self.LeakyReLU(x)
        x = self.Conv2d44(x)
        x = self.BatchNorm2d4(x)
        x = self.LeakyReLU(x)
        x = self.Conv2d48(x)
        # x = self.BatchNorm2d8(x)
        # x = self.LeakyReLU(x)
        # x = self.Conv2dEnd(x)
        x = self.Flatten(x)
        x = self.Sigmoid(x)
        return x
