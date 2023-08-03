import torch
from torch import nn

from src.model import RestNet_utilis, DesNet_utilis
from src.model import VGG_utilis
from src.model.AlexNet import AlexNet as an
# from src.model.ACGan import Generator as g
from src.model.Generator import GeneratorResnet as g
from src.model.Generator import LstmRNN as l
# from src.model.Generator import ComplexLSTM as cl
from src.model.InferenceAttack import InferenceAttack_HZ


class AlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, *args, **kwargs):
        super(AlexNet, self).__init__()
        self.moedl = an(in_channels=in_channels, num_classes=num_classes)

    def forward(self, x):
        return self.moedl(x)


class ResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, *args, **kwargs):
        super(ResNet, self).__init__()
        self.model = RestNet_utilis.ResNet18(in_channels=in_channels, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


class DesNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, *args, **kwargs):
        super(DesNet, self).__init__()
        self.model = DesNet_utilis.densenet121(in_channels=in_channels, num_class=num_classes)

    def forward(self, x):
        return self.model(x)


class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, *args, **kwargs):
        super(VGG, self).__init__()
        self.model = VGG_utilis.VGG(vgg_name='vgg16', in_channels=in_channels, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3):
        super(Generator, self).__init__()
        self.model = g(in_channels=in_channels)

    def forward(self, x):
        return self.model(x)

class LinearGenerator(nn.Module):
    def __init__(self, in_features=3):
        super(LinearGenerator, self).__init__()
        self.model  = nn.Sequential(
            # nn.Linear(in_features, 2048),
            nn.Linear(in_features, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, in_features)
        )

    def forward(self, x):
        return self.model(x)

class InferenceAttack(nn.Module):
    def __init__(self, num_classes=10):
        super(InferenceAttack, self).__init__()
        self.model = InferenceAttack_HZ(num_classes=num_classes)

    def forward(self, attack_input, infer_input_one_hot):
        return self.model(attack_input, infer_input_one_hot)

# NN for Purchase100
class NN(nn.Module):
    def __init__(self, num_classes=100, in_features=600, *args, **kwargs):
        super(NN, self).__init__()
        self.layers = nn.Sequential(
            # nn.Linear(in_features, 1024),
            nn.Linear(in_features, 512),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

# NN for Texas100
class NN_5(nn.Module):
    def __init__(self, num_classes=100, in_features=600, *args, **kwargs):
        super(NN_5, self).__init__()
        self.layers = nn.Sequential(
            # nn.Linear(in_features, 2048),
            nn.Linear(in_features, 1024),
            nn.Linear(1024,512),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    model = LinearGenerator(in_features=6169)
    x = torch.rand(1,6169)
    print(model(x).shape)
