from __future__ import print_function

from models import *
from utils.denoising_utils import *
from utils.wandb_utils import *
import torch
import torch.optim
import matplotlib.pyplot as plt

import os
import wandb
import argparse
import numpy as np
# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--gpu', default='0')
parser.add_argument('--index', default=0, type=int)
parser.add_argument('--input_index', default=0, type=int)
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
imsize = -1
PLOT = True
sigma = 25
sigma_ = sigma/255.
train_input = False

fnames = ['data/denoising/F16_GT.png', 'data/inpainting/kate.png', 'data/inpainting/vase.png', 'data/sr/zebra_GT.png']
fname = fnames[args.index]

if fname == 'data/denoising/snail.jpg':
    img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_noisy_np = pil_to_np(img_noisy_pil)

    # As we don't have ground truth
    img_pil = img_noisy_pil
    img_np = img_noisy_np

    if PLOT:
        plot_image_grid([img_np], 4, 5)

elif fname in fnames:
    # Add synthetic noise
    img_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_np = pil_to_np(img_pil)

    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

    # if PLOT:
    #     plot_image_grid([img_np, img_noisy_np], 4, 6)
else:
    assert False


INPUT = ['noise', 'fourier', 'meshgrid', 'infer_freqs'][args.input_index]
pad = 'reflection'
OPT_OVER = 'net'  # 'net,input'

reg_noise_std = 1. / 30. if INPUT == 'noise' else 0.  # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER = 'adam'  # 'LBFGS'
show_every = 100
exp_weight = 0.99

if fname == 'data/denoising/snail.jpg':
    num_iter = 2400
    input_depth = 3
    figsize = 5

    net = skip(
        input_depth, 3,
        num_channels_down=[8, 16, 32, 64, 128],
        num_channels_up=[8, 16, 32, 64, 128],
        num_channels_skip=[0, 0, 0, 4, 4],
        upsample_mode='bilinear',
        need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(dtype)

elif fname in fnames:
    num_iter = 10000
    input_depth = 32
    figsize = 4

    net = get_net(input_depth, 'skip', pad,
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)

else:
    assert False

# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()]);
print('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

out_avg = None
last_net = None
psrn_noisy_last = 0
psnr_gt_list = []
i = 0
n_freqs = 8
penet = PENet(num_frequencies=8, img_size=(img_pil.size[1], img_pil.size[0])).type(dtype)
freq_input_saved = torch.rand(8).detach().type(dtype)


def closure():
    global i, out_avg, psrn_noisy_last, last_net, psnr_gt_list, freq_input_saved

    net_input = penet(freq_input_saved)
    out = net(net_input)

    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    total_loss = mse(out, img_noisy_torch)
    total_loss.backward()

    psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
    psrn_gt = compare_psnr(img_np, out.detach().cpu().numpy()[0])
    psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])

    # Note that we do not have GT for the "snail" example
    # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    if PLOT and i % show_every == 0:
        print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (
            i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm))
        psnr_gt_list.append(psrn_gt)
        if train_input:
            log_inputs(net_input)

        wandb.log({'psnr_gt': psrn_gt, 'psnr_noisy': psrn_noisy}, commit=False)

    # Backtracking
    if i % show_every:
        if psrn_noisy - psrn_noisy_last < -5:
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return total_loss * 0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_noisy_last = psrn_noisy

    i += 1

    wandb.log({'training loss': total_loss.item()}, commit=True)
    return total_loss


log_config = {
    "learning_rate": LR,
    "epochs": num_iter,
    'optimizer': OPTIMIZER,
    'loss': type(mse).__name__,
    'input depth': input_depth,
    'input type': INPUT,
    'Train input': train_input
}
run = wandb.init(project="Fourier features DIP",
                 entity="impliciteam",
                 tags=['infer_freqs', 'depth:{}'.format(input_depth)],
                 name='{}_depth_{}_{}'.format(os.path.basename(fname).split('.')[0], input_depth, INPUT),
                 job_type='train',
                 group='Denoising',
                 mode='online',
                 save_code=True,
                 config=log_config,
                 notes='Input type {} - {} random projected to depth {}'.format(
                     INPUT, n_freqs, input_depth))

wandb.run.log_code(".")
p = get_params(OPT_OVER, net, freq_input_saved)
optimize(OPTIMIZER, p, closure, LR, num_iter)

out_np = torch_to_np(net(penet(freq_input_saved)))
log_images(np.array([np.clip(out_np, 0, 1), img_np]), num_iter, task='Denoising')
q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13)
plt.plot(psnr_gt_list)
plt.title('max: {}\nlast: {}'.format(max(psnr_gt_list), psnr_gt_list[-1]))
plt.show()
