from __future__ import print_function
import random
import argparse

from models import *
from utils.denoising_utils import *

import torch.optim
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

parser = argparse.ArgumentParser()
parser.add_argument('--input_img_path', default='./data/F16_GT.png')
args = parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

# Fix seeds
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

imsize = -1
PLOT = True
sigma = 25
sigma_ = sigma/255.

fnames = [args.input_img_path]
fname = fnames[0]

if fname in fnames:
    img_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_np = pil_to_np(img_pil)
    output_depth = img_np.shape[0]

    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

    if PLOT:
        plot_image_grid([img_np, img_noisy_np], 4, 6)
else:
    assert False

INPUT = 'fourier'
pad = 'reflection'
OPT_OVER = 'net'

train_input = True if ',' in OPT_OVER else False
reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER = 'adam'  # 'LBFGS'
show_every = 100
exp_weight = 0.99

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

if fname in fnames:
    num_iter = 1800
    figsize = 4
    freq_dict = {
        'method': 'log',
        'cosine_only': False,
        'n_freqs': 8,
        'base': 2 ** (8 / (8 - 1)),
    }
    input_depth = freq_dict['n_freqs'] * 4

    net = get_net(input_depth, 'skip', pad, n_channels=output_depth,
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)

else:
    assert False

net_input = get_input(input_depth, INPUT, (img_pil.size[1], img_pil.size[0]), freq_dict=freq_dict).type(dtype)

# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()])

# Loss
mse = torch.nn.MSELoss().type(dtype)


net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

out_avg = None
last_net = None
psrn_noisy_last = 0
psnr_gt_list = []
i = 0


def closure():
    global i, out_avg, psrn_noisy_last, last_net, net_input, psnr_gt_list

    if INPUT == 'noise':
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        else:
            net_input = net_input_saved
    elif INPUT == 'fourier':
        net_input = net_input_saved
    elif INPUT == 'infer_freqs':
        if reg_noise_std > 0:
            net_input_ = net_input_saved + (noise.normal_() * reg_noise_std)
        else:
            net_input_ = net_input_saved

        net_input = generate_fourier_feature_maps(net_input_,  (img_pil.size[1], img_pil.size[0]), dtype)
    else:
        net_input = net_input_saved

    out = net(net_input)
    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    total_loss = mse(out, img_noisy_torch)
    total_loss.backward()

    out_np = out.detach().cpu().numpy()[0]
    psrn_noisy = compare_psnr(img_noisy_np, out_np)
    psrn_gt = compare_psnr(img_np, out_np)
    psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])

    if PLOT and i % show_every == 0:
        print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (
            i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm))
        psnr_gt_list.append(psrn_gt)

    # Backtracking
    # if i % show_every:
        if psrn_noisy - psrn_noisy_last < -2:
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return total_loss * 0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_noisy_last = psrn_noisy

    i += 1

    # Log metrics
    return total_loss

print('Number of params: %d' % s)
print(net)
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

net_input = net_input_saved

out_np = torch_to_np(net(net_input))
q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13)
filename = os.path.basename(fname).split('.')[0]
plt.imsave('denoising_fixed_ff.png', out_np.transpose(1, 2, 0))
plt.show()
