import torch.nn as nn
import torch.nn.functional as F


class ReLUNet(nn.Module):
    def __init__(self, H=200, d=2, bw=True):
        super(ReLUNet, self).__init__()
        out_dim = (1 if bw else 3)
        self.w = [nn.Linear(2, H, bias=True)] + \
                 [nn.Linear(H, H, bias=True) for _ in range(d - 2)] + \
                 [nn.Linear(H, out_dim, bias=True)]

        for i, wi in enumerate(self.w):
            self.add_module('w' + str(i), wi)

    def forward(self, x):
        out = x.flatten()
        d = len(self.w)
        for i in range(d - 1):
            out = self._modules['w' + str(i)](out)
            out = F.relu(out, inplace=True)
        out = self._modules['w' + str(d - 1)](out)
        out = F.sigmoid(out)
        return out