import torch.nn as nn

# 用以提供分类训练的基础模型Softmax
class Softmax_Model(nn.Module):

    def __init__(self, n_in, n_out):

        super(Softmax_Model, self).__init__()

        self.fc1 = nn.Sequential(
                            nn.Linear(in_features=n_in,
                                      out_features=n_out)
                )

        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = self.fc1(x)
        return x


