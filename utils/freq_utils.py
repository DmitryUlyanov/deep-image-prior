import torch
from torch.fft import *
import matplotlib.pyplot as plt


def analyze_frequencies(img_var):
    img_f = rfft2(img_var, norm="ortho")
    mag_img_f = torch.abs(img_f).cpu()
    img_shifted_f = fftshift(img_f, dim=[-2])
    mag_shifted_f = torch.abs(img_shifted_f.cpu())
    plt.imshow(mag_shifted_f[0].permute(1, 2, 0))
    plt.show()
    hist = torch.histogram(mag_img_f)
    plt.bar(hist.bin_edges[:-1], hist.hist)
    plt.ylim([0, 150])
    plt.show()
    print('max frequency range: {}'.format(max(hist.bin_edges)))

