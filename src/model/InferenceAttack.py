import torch
import torch.nn as nn


class InferenceAttack_HZ(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(InferenceAttack_HZ, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(num_classes, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )
        self.labels = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(64 * 2, 512),

            nn.ReLU(),
            nn.Linear(512, 256),

            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                nn.init.normal_(self.state_dict()[key], std=0.01)

            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output = nn.Sigmoid()

    def forward(self, x1, l):
        # print (l.size(),x.size())
        out_x1 = self.features(x1)

        out_l = self.labels(l)

        is_member = self.combine(torch.cat((out_x1, out_l), 1))

        return self.output(is_member)
