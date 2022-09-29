import torch
from torch.fft import fftshift, rfft2
import pywt
import matplotlib.pyplot as plt


def analyze_frequencies(img_var, resolution_factor=2):
    import matplotlib.colors as mcolors

    img_f = rfft2(img_var)
    mag_img_f = torch.abs(img_f).cpu()
    img_shifted_f = fftshift(img_f, dim=[-2])
    mag_shifted_f = torch.abs(img_shifted_f.cpu())
    plt.imshow(mag_shifted_f[0].permute(1, 2, 0))
    plt.colorbar()
    plt.show()
    bins = torch.Tensor([torch.Tensor([0]), * list(2 ** torch.linspace(0, 6, 7))]) / resolution_factor
    hist = torch.histogram(mag_img_f, bins=bins)
    max_freq = hist.bin_edges[int(max(torch.nonzero(hist.hist, as_tuple=True)[0]))]
    print(hist.hist)
    print(bins)
    plt.bar(hist.bin_edges[:-1], hist.hist, width=0.5, color=[k for k in mcolors.BASE_COLORS.keys()])
    plt.ylim([0, 150])
    plt.xlim([0, max_freq])
    plt.show()
    print('max frequency: {}'.format(max_freq.item()))


def visualize_learned_frequencies(learned_frequencies):
    import wandb
    [wandb.log({'learned Frequency #{}'.format(i): learned_frequencies[i]}, commit=False)
     for i in range(learned_frequencies.shape[0])]


def analyze_image(img_torch, size):
    w = pywt.Wavelet('db3')
    size = size  # patch size
    stride = size  # patch stride

    patches = img_torch.unfold(2, size, stride).unfold(3, size, stride).cpu()
    wt_list_cols = []
    for ver_idx in range(patches.shape[2]):
        wt_list_rows = []
        for hor_idx in range(patches.shape[3]):
            current_patch = patches[0, :, ver_idx, hor_idx, :, :]
            wt_list_rows.append(pywt.swt2(current_patch.numpy(), w, level=2))
        wt_list_cols.append(wt_list_rows)
    pass