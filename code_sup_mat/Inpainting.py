from __future__ import print_function

from utils.common_utils import *
import matplotlib.pyplot as plt
import os
import numpy as np
from models.skip import skip
import torch.optim
import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

# Fix seeds
seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--input_img_path', default='./data/F16_GT.png')
parser.add_argument('--input_mask_path', default='./data/mask_image_F16_512rgb.png')
args = parser.parse_args()

PLOT = True
imsize = -1
dim_div_by = 64

INPUT = 'infer_freqs'
img_path = args.input_img_path
mask_path = args.input_mask_path

img_pil, img_np = get_image(img_path, imsize)
img_mask_pil, img_mask_np = get_image(mask_path, imsize)

img_mask_pil = crop_image(img_mask_pil, dim_div_by)
img_pil = crop_image(img_pil, dim_div_by)

img_np = pil_to_np(img_pil)
img_mask_np = pil_to_np(img_mask_pil)

img_mask_var = np_to_torch(img_mask_np).type(dtype)

plot_image_grid([img_np, img_mask_np, img_mask_np * img_np], 3, 11)

freq_dict = {
        'method': 'log',
        'cosine_only': False,
        'n_freqs': 8,
        'base': 2 ** (8 / (8-1))
    }
pad = 'reflection'  # 'zero'
OPTIMIZER = 'adam'
input_depth = freq_dict['n_freqs'] * 4
param_noise = False
show_every = 50
figsize = 5
reg_noise_std = 0.03
LR = 0.01

if 'vase.png' in img_path:
    num_iter = 8001
    net = skip(input_depth, img_np.shape[0],
               num_channels_down=[128] * 5,
               num_channels_up=[128] * 5,
               num_channels_skip=[4] * 5,
               filter_size_up=1, filter_size_down=1, filter_skip_size=1,
               upsample_mode='bilinear',  # downsample_mode='avg',
               need1x1_up=True,
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

else:
    num_iter = 3000

    net = skip(input_depth, img_np.shape[0],
               num_channels_down=[128] * 6,
               num_channels_up=[128] * 6,
               num_channels_skip=[4] * 6,
               filter_size_up=1, filter_size_down=1,
               upsample_mode='bilinear', filter_skip_size=1,
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

net = net.type(dtype)
net_input = get_input(input_depth, INPUT, (img_pil.size[1], img_pil.size[0]), freq_dict=freq_dict).type(dtype)
noise = net_input.detach().clone()

if INPUT == 'infer_freqs':
    OPT_OVER = 'net,input'
else:
    OPT_OVER = 'net'

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_var = np_to_torch(img_np).type(dtype)
mask_var = np_to_torch(img_mask_np).type(dtype)

psnr_masked_last = 0.0
last_net = None
best_psnr_gt = -1.0
best_iter = 0
i = 0


def closure():
    global i, last_net, best_psnr_gt, best_iter, psnr_masked_last

    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std() / 50

    if reg_noise_std > 0:
        net_input_ = net_input_saved + (noise.normal_() * reg_noise_std)
    else:
        net_input_ = net_input_saved

    net_input = generate_fourier_feature_maps(net_input_, (img_pil.size[1], img_pil.size[0]), dtype,
                                              only_cosine=freq_dict['cosine_only'])

    out = net(net_input)

    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()

    if PLOT and i % show_every == 0:
        out_np = out.detach().cpu().numpy()[0]
        psnr_gt = compare_psnr(img_np, out_np)
        psnr_masked = compare_psnr(img_np * img_mask_np, out_np)

        print('Iteration %05d    Loss %f    psnr_gt %f   psnr_masked %f' % (i, total_loss.item(), psnr_gt, psnr_masked))
        if psnr_gt > best_psnr_gt:
            best_psnr_gt = psnr_gt
            best_iter = i

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


filename = os.path.basename(img_path).split('.')[0]
# Compute number of parameters
s = sum(np.prod(list(p.size())) for p in net.parameters())
print('Number of params: %d' % s)
print(net)
net_input_saved = net_input
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)
net_input = generate_fourier_feature_maps(net_input_saved, (img_pil.size[1], img_pil.size[0]), dtype,
                                          only_cosine=freq_dict['cosine_only'])

out_np = torch_to_np(net(net_input))
plot_image_grid([out_np, img_np], factor=5)
filename = os.path.basename(img_path).split('.')[0]
plt.imsave('{}_inpainting_learned_ff.png'.format(filename), out_np.transpose(1, 2, 0))
plt.show()
