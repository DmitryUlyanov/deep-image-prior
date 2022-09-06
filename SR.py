from __future__ import print_function
import matplotlib.pyplot as plt

import argparse
import os
import tqdm
import numpy as np
from models import *

import torch
import torch.optim

# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from models.downsampler import Downsampler

from utils.sr_utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize = -1
factor = 4 # 8
enforse_div32 = 'CROP' # we usually need the dimensions to be divisible by a power of two (32 in this case)
PLOT = True
show_every = 100

# To produce images from the paper we took *_GT.png images from LapSRN viewer for corresponding factor,
# e.g. x4/zebra_GT.png for factor=4, and x8/zebra_GT.png for factor=8
path_to_image = 'data/sr/zebra_GT.png'

# Starts here
imgs = load_LR_HR_imgs_sr(path_to_image, imsize, factor, enforse_div32)

imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(imgs['LR_pil'], imgs['HR_pil'])

if PLOT:
    # plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np']], 4,12);
    print('PSNR bicubic: %.4f   PSNR nearest: %.4f' %  (
                                        compare_psnr(imgs['HR_np'], imgs['bicubic_np']),
                                        compare_psnr(imgs['HR_np'], imgs['nearest_np'])))

input_depth = 32

INPUT = ['meshgrid', 'noise', 'fourier'][-1]
pad = 'reflection'
OPT_OVER = 'net'
KERNEL_TYPE = 'lanczos2'
sample_freqs = True
LR = 0.01
tv_weight = 0.0

OPTIMIZER = 'adam'

if factor == 4:
    num_iter = 50000
    reg_noise_std = 0  # 0.03
elif factor == 8:
    num_iter = 4000
    reg_noise_std = 0.05
else:
    assert False, 'We did not experiment with other factors'

net_input = get_noise(input_depth, INPUT, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0]))

print('Input is {}, Depth = {}'.format(INPUT, input_depth))
net_input = net_input.type(dtype).detach()

NET_TYPE = 'skip' # UNet, ResNet
net = get_net(input_depth, 'skip', pad,
              skip_n33d=128,
              skip_n33u=128,
              skip_n11=4,
              num_scales=5,
              upsample_mode='bilinear').type(dtype)

# Losses
mse = torch.nn.MSELoss().type(dtype)

img_LR_var = np_to_torch(imgs['LR_np']).type(dtype)

downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)


def closure():
    global i, net_input, last_net, psnr_LR_last, indices

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    if sample_freqs:
        # if i % 8000 == 0:  # sample freq
        if 1:
            indices = torch.multinomial(torch.arange(0, net_input_saved.size(1), dtype=torch.float),
                                        input_depth, replacement=False)

            assert len(torch.unique(indices)) == input_depth
            # print(indices)

        net_input = net_input_saved[:, indices, :, :]

    out_HR = net(net_input)
    out_LR = downsampler(out_HR)

    total_loss = mse(out_LR, img_LR_var)

    if tv_weight > 0:
        total_loss += tv_weight * tv_loss(out_HR)

    total_loss.backward()

    # Log
    psnr_LR = compare_psnr(imgs['LR_np'], torch_to_np(out_LR))
    psnr_HR = compare_psnr(imgs['HR_np'], torch_to_np(out_HR))

    # if psnr_LR - psnr_LR_last < -5:
    #     print('Falling back to previous checkpoint.')
    #
    #     for new_param, net_param in zip(last_net, net.parameters()):
    #         net_param.data.copy_(new_param.cuda())
    #
    #     return total_loss * 0
    # else:
    #     last_net = [x.detach().cpu() for x in net.parameters()]
    #     psnr_LR_last = psnr_LR

    # History
    psnr_history.append([psnr_LR, psnr_HR])

    if PLOT and i % 100 == 0:
        print('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f' % (i, psnr_LR, psnr_HR))
        # print(indices)
        # out_HR_np = torch_to_np(out_HR)
        # plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], np.clip(out_HR_np, 0, 1)], factor=13, nrow=3)

    i += 1

    return total_loss


# %%

psnr_history = []
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
indices = torch.arange(0, input_depth, dtype=torch.float)
last_net = None
psnr_LR_last = 0

i = 0
p = get_params(OPT_OVER, net, net_input)

optimizer = torch.optim.Adam(p, lr=LR)
for j in tqdm.tqdm(range(num_iter)):
    optimizer.zero_grad()
    closure()
    optimizer.step()

# optimize(OPTIMIZER, p, closure, LR, num_iter)

out_HR_np = np.clip(torch_to_np(net(net_input)), 0, 1)
result_deep_prior = put_in_center(out_HR_np, imgs['orig_np'].shape[1:])

# For the paper we acually took `_bicubic.png` files from LapSRN viewer and used `result_deep_prior` as our result
plot_image_grid([imgs['HR_np'],
                 imgs['bicubic_np'],
                 out_HR_np], factor=4, nrow=1)

fig, axes = plt.subplots(1, 2)
axes[0].plot([h[0] for h in psnr_history])
axes[0].set_title('LR PSNR\nmax: {:.3f}'.format(max([h[0] for h in psnr_history])))
axes[1].plot([h[1] for h in psnr_history])
axes[1].set_title('HR PSNR\nmax: {:.3f}'.format(max([h[1] for h in psnr_history])))
plt.savefig('curve_{}.png'.format(num_iter))
