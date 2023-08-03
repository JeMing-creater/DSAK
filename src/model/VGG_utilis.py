from __future__ import print_function
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

# vgg网络模型配置列表，数字表示卷积核个数，'M'表示最大池化层

class VGG(nn.Module):
    def __init__(self, vgg_name='vgg11',in_channels = 3,num_classes=10, init_weights=False):#
        super(VGG, self).__init__()
        cfgs = {
                'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # 模型A
                'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # 模型B
                'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                # 模型D
                'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512,
                          512, 'M'],  # 模型E
                }  # 输入的原始图像(rgb三通道)
        self.features = self.make_features(in_channels,cfgs[vgg_name])			# 卷积层提取特征
        self.classifier = nn.Sequential(
                # 14
                nn.Linear(512, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                # 15
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                # 16
                nn.Linear(4096, num_classes),
                )
        if init_weights:
            self._initialize_weights()

    # 卷积层提取特征
    def make_features(self,in_channels,cfg: list):  # 传入的是具体某个模型的参数列表
        layers = []
        in_channel = in_channels
        for v in cfg:
            # 最大池化层
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # 卷积层
            else:
                conv2d = nn.Conv2d(in_channel, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(True)]
                in_channel = v
        return nn.Sequential(*layers)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    # 实例化例子
    # 通过vgg_name指定
    net = VGG(vgg_name='vgg19',in_channels=1,num_classes=10).cuda()

    mnist_train = datasets.MNIST(
        '/kolla/jeming/MEA_Evaluation/datasets', True, transform=transforms.Compose(
                [
                        transforms.ToTensor(),
                        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
                        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
                        transforms.Normalize(
                            mean=[0.485],
                            std=[0.229]
                            )
                        ]
                ), download=True
        )

    mnist_test = datasets.MNIST(
        '/kolla/jeming/MEA_Evaluation/datasets', False, transform=transforms.Compose(
                [
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485],
                            std=[0.229]
                            )
                        ]
                ), download=True
        )

    train = DataLoader(mnist_train, batch_size=100, shuffle=True)
    test = DataLoader(mnist_test, batch_size=100, shuffle=True)
    loss_func2 = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for step, (xx, label) in enumerate(test):
        if torch.cuda.is_available():
            xx = xx.cuda()
            label = label.cuda()
        if isinstance(xx, nn.AvgPool2d):
            xx.ceil_mode = True
        prelabel = net(xx)
        loss = loss_func2(prelabel, label)
        print('yes')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


