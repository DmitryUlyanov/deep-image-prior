import torch
import torch.nn as nn
import numpy as np


class PENet(nn.Module):
    def __init__(self, num_frequencies, img_size, dtype=torch.cuda.FloatTensor):
        super(PENet, self).__init__()
        self.img_size = img_size
        X, Y = np.meshgrid(np.arange(0, self.img_size[1]) / float(self.img_size[1] - 1),
                           np.arange(0, self.img_size[0]) / float(self.img_size[0] - 1))
        meshgrid_np = np.concatenate([X[None, :], Y[None, :]])
        self.meshgrid = torch.from_numpy(meshgrid_np).permute(1, 2, 0).unsqueeze(0).type(dtype)
        self.num_freqs = num_frequencies
        self.linear1 = nn.Linear(num_frequencies, 100)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(100, num_frequencies)

    def forward(self, x):
        # j = torch.arange(m, device=x.device)
        # coeffs = 2 * np.pi * sigma ** (j / m)
        x = self.linear1(x)
        x = self.relu(x)
        coeffs = self.linear2(x)
        vp = coeffs * torch.unsqueeze(self.meshgrid, -1)
        vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
        return vp_cat.flatten(-2, -1).permute(0, 3, 1, 2)


# n_freqs = 10
# penet = PENet(num_frequencies=n_freqs, img_size=(512, 512))
# loss = nn.MSELoss()
# optimizer = torch.optim.Adam(penet.parameters(), lr=0.01)
# x = torch.randn(n_freqs).clone().detach()
# for i in range(1000):
#     optimizer.zero_grad()
#     res = penet(x)
#     loss_ = loss(res, torch.ones_like(res))
#     if i % 10 == 0:
#         print('iter #{}: {}'.format(i, loss_.item()))
#     loss_.backward()
#     optimizer.step()
#
#
