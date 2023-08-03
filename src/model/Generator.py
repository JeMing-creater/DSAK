import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

###########################
# Generator: Resnet
###########################

# To control feature map in generator
ngf = 64


class GeneratorResnet(nn.Module):
    def __init__(self, in_channels=3, for_Linear=False, inception=False):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        '''
        super(GeneratorResnet, self).__init__()
        # Input_size = 3, n, n
        self.inception = inception
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        self.resblock3 = ResidualBlock(ngf * 4)
        self.resblock4 = ResidualBlock(ngf * 4)
        self.resblock5 = ResidualBlock(ngf * 4)
        self.resblock6 = ResidualBlock(ngf * 4)

        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, in_channels, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)
        return (torch.tanh(x) + 1) / 2 + input  # Output range [0 1]

class BiRNN(nn.Module):
    def __init__(self, in_features=1, hidden_size=2048):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(in_features, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, in_features)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        hidden = self.init_hidden(batch_size)
        inputs = inputs.permute(1, 0, 2)  # 将 batch 维度放在第二维
        output, hidden = self.rnn(inputs, hidden)
        # 将前向和后向的输出拼接在一起
        output = torch.cat((output[:, :, :self.hidden_size], output[:, :, self.hidden_size:]), dim=-1)
        output = self.fc(output)
        output = self.softmax(output)
        return output

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size)


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual


if __name__ == '__main__':
    netG = GeneratorResnet(in_channels=1)
    test_sample = torch.rand(1, 6169)
    print('Generator output:', netG(test_sample).size())
    print('Generator parameters:', sum(p.numel() for p in netG.parameters() if p.requires_grad))
