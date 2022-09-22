import torch.nn as nn


class FCN(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Conv2d(lastv, hidden, kernel_size=(3, 3), padding='same'))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Conv2d(lastv, out_dim, kernel_size=(3, 3), padding='same'))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

