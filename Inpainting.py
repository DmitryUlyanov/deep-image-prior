from __future__ import print_function

from utils.inpainting_utils import *

import matplotlib.pyplot as plt
import os
import numpy as np
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
import torch
import torch.optim
import wandb
from skimage.metrics import peak_signal_noise_ratio as compare_psnr



torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
PLOT = True
imsize = -1
dim_div_by = 64

## Fig 6
# img_path  = 'data/inpainting/vase.png'
# mask_path = 'data/inpainting/vase_mask.png'

## Fig 8
img_path  = 'data/inpainting/library.png'
mask_path = 'data/inpainting/library_mask.png'

## Fig 7 (top)
# img_path = 'data/inpainting/kate.png'
# mask_path = 'data/inpainting/kate_mask.png'

# Another text inpainting example
# img_path  = 'data/inpainting/peppers.png'
# mask_path = 'data/inpainting/peppers_mask.png'

NET_TYPE = 'skip_depth6'  # one of skip_depth4|skip_depth2|UNET|ResNet

img_pil, img_np = get_image(img_path, imsize)
img_mask_pil, img_mask_np = get_image(mask_path, imsize)

img_mask_pil = crop_image(img_mask_pil, dim_div_by)
img_pil = crop_image(img_pil, dim_div_by)

img_np = pil_to_np(img_pil)
img_mask_np = pil_to_np(img_mask_pil)

img_mask_var = np_to_torch(img_mask_np).type(dtype)

plot_image_grid([img_np, img_mask_np, img_mask_np * img_np], 3, 11);

pad = 'reflection'  # 'zero'
OPT_OVER = 'net'
OPTIMIZER = 'adam'

if 'vase.png' in img_path:
    INPUT = 'meshgrid'
    input_depth = 2
    LR = 0.01
    num_iter = 5001
    param_noise = False
    show_every = 50
    figsize = 5
    reg_noise_std = 0.03

    net = skip(input_depth, img_np.shape[0],
               num_channels_down=[128] * 5,
               num_channels_up=[128] * 5,
               num_channels_skip=[0] * 5,
               upsample_mode='nearest', filter_skip_size=1, filter_size_up=3, filter_size_down=3,
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

elif ('kate.png' in img_path) or ('peppers.png' in img_path):
    # Same params and net as in super-resolution and denoising
    INPUT = 'noise' #'fourier'
    input_depth = 32
    LR = 0.01
    num_iter = 5001
    param_noise = False
    show_every = 50
    figsize = 5
    reg_noise_std = 0.03

    net = skip(input_depth, img_np.shape[0],
               num_channels_down=[128] * 5,
               num_channels_up=[128] * 5,
               num_channels_skip=[128] * 5,
               filter_size_up=3, filter_size_down=3,
               upsample_mode='nearest', filter_skip_size=1,
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

elif 'library.png' in img_path:

    INPUT = 'fourier' #'noise'
    input_depth = 32

    num_iter = 5001
    show_every = 50
    figsize = 8
    reg_noise_std = 0.00
    param_noise = True

    if 'skip' in NET_TYPE:

        depth = int(NET_TYPE[-1])
        net = skip(input_depth, img_np.shape[0],
                   num_channels_down=[16, 32, 64, 128, 128, 128][:depth],
                   num_channels_up=[16, 32, 64, 128, 128, 128][:depth],
                   num_channels_skip=[0, 0, 0, 0, 0, 0][:depth],
                   filter_size_up=3, filter_size_down=5, filter_skip_size=1,
                   upsample_mode='nearest',  # downsample_mode='avg',
                   need1x1_up=False,
                   need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

        LR = 0.01

    elif NET_TYPE == 'UNET':

        net = UNet(num_input_channels=input_depth, num_output_channels=3,
                   feature_scale=8, more_layers=1,
                   concat_x=False, upsample_mode='deconv',
                   pad='zero', norm_layer=torch.nn.InstanceNorm2d, need_sigmoid=True, need_bias=True)

        LR = 0.001
        param_noise = False

    elif NET_TYPE == 'ResNet':

        net = ResNet(input_depth, img_np.shape[0], 8, 32, need_sigmoid=True, act_fun='LeakyReLU')

        LR = 0.001
        param_noise = False

    else:
        assert False
else:
    assert False

net = net.type(dtype)
net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)

# Compute number of parameters
s = sum(np.prod(list(p.size())) for p in net.parameters())
print('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_var = np_to_torch(img_np).type(dtype)
mask_var = np_to_torch(img_mask_np).type(dtype)
psnr_gt_list = []
psnr_mask_list = []
psnr_masked_last = 0.0
last_net = None
indices = torch.arange(0, input_depth, dtype=torch.float)
sample_freqs = True if INPUT == 'fourier' else False
i = 0


def closure():
    global i, indices, last_net, psnr_masked_last

    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std() / 50

    net_input = net_input_saved
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    if sample_freqs:
        if i % num_iter == 0:  # sample freq
            indices = torch.multinomial(torch.arange(0, net_input_saved.size(1), dtype=torch.float),
                                        input_depth, replacement=False)

            assert len(torch.unique(indices)) == input_depth
            print(indices)

        net_input = net_input_saved[:, indices, :, :]

    out = net(net_input)

    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()

    if PLOT and i % show_every == 0:
        psnr_gt = compare_psnr(img_np, out.detach().cpu().numpy()[0])
        psnr_masked = compare_psnr(img_np * img_mask_np, out.detach().cpu().numpy()[0])
        psnr_gt_list.append(psnr_gt)
        psnr_mask_list.append(psnr_masked)
        print('Iteration %05d    Loss %f    psnr_gt %f   psnr_masked %f' % (i, total_loss.item(), psnr_gt, psnr_masked))
        # plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)

    if i % show_every == 0:
        if psnr_masked - psnr_masked_last < -1:
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return total_loss * 0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psnr_masked_last = psnr_masked

    i += 1

    return total_loss


net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

net_input = net_input_saved[:, indices, :, :] if sample_freqs else net_input
out_np = torch_to_np(net(net_input))
plot_image_grid([out_np, img_np], factor=5)
plt.plot(psnr_gt_list)
plt.title('max: {}\nlast: {}'.format(max(psnr_gt_list), psnr_gt_list[-1]))
plt.show()
