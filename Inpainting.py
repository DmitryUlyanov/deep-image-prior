from __future__ import print_function

from utils.inpainting_utils import *
from utils.wandb_utils import *
from models import PENet
import matplotlib.pyplot as plt
import os
import numpy as np
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
import torch
import torch.optim
import wandb
import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

os.environ['WANDB_IGNORE_GLOBS'] = './venv/**/*.*'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

# Fix seeds
seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--gpu', default='1')
parser.add_argument('--index', default=0, type=int)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
PLOT = True
imsize = -1
dim_div_by = 64


img_dict = {
    0: {  # Fig 6
            'img_path': 'data/inpainting/vase.png',
            'mask_path': 'data/inpainting/vase_mask.png'
    },
    1: {  # Fig 7 (top)
        'img_path': 'data/inpainting/kate.png',
        'mask_path': 'data/inpainting/kate_mask.png'
    },
    2: {  # Fig 8
        'img_path': 'data/inpainting/library.png',
        'mask_path': 'data/inpainting/library_mask.png'
    },
    3: {
        'img_path': 'data/inpainting/peppers.png',
        'mask_path': 'data/inpainting/peppers_mask.png'

    }
}

img_path = img_dict[args.index]['img_path']
mask_path = img_dict[args.index]['mask_path']
NET_TYPE = 'skip_depth6'  # one of skip_depth4|skip_depth2|UNET|ResNet

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
        'max': 64,
        'n_freqs': 8
    }
pad = 'reflection'  # 'zero'
OPT_OVER = 'net'
OPTIMIZER = 'adam'
train_input = True if ',' in OPT_OVER else False

if 'vase.png' in img_path:
    INPUT = 'fourier'  # 'meshgrid' # 'infer_freqs'
    input_depth = 32
    LR = 0.01
    num_iter = 8001
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
    INPUT = 'infer_freqs'  #'infer_freqs'  # 'noise'
    input_depth = 32
    LR = 0.01
    num_iter = 6001
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

    INPUT = 'fourier'  # 'noise'  # 'infer_freqs'
    input_depth = 32

    num_iter = 8001
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
net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0]), freq_dict=freq_dict).type(dtype)
noise = net_input.detach().clone()

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
i = 0


def closure():
    global i, last_net, psnr_masked_last

    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std() / 50

    if INPUT == 'noise':
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        else:
            net_input = net_input_saved
    elif INPUT == 'fourier':
        # net_input = net_input_saved[:, indices, :, :]
        net_input = net_input_saved
    elif INPUT == 'infer_freqs':
        if reg_noise_std > 0:
            net_input_ = net_input_saved + (noise.normal_() * reg_noise_std)
        else:
            net_input_ = net_input_saved
        net_input = generate_fourier_feature_maps(net_input_, (img_pil.size[1], img_pil.size[0]), dtype)
    else:
        net_input = net_input_saved

    out = net(net_input)

    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()

    if PLOT and i % show_every == 0:
        psnr_gt = compare_psnr(img_np, out.detach().cpu().numpy()[0])
        psnr_masked = compare_psnr(img_np * img_mask_np, out.detach().cpu().numpy()[0])
        psnr_gt_list.append(psnr_gt)
        psnr_mask_list.append(psnr_masked)
        if train_input:
            log_inputs(net_input)

        print('Iteration %05d    Loss %f    psnr_gt %f   psnr_masked %f' % (i, total_loss.item(), psnr_gt, psnr_masked))
        wandb.log({'psnr_gt': psnr_gt, 'psnr_noisy': psnr_masked}, commit=False)
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

    wandb.log({'training loss': total_loss.item()}, commit=True)
    return total_loss


log_config = {
    "learning_rate": LR,
    "epochs": num_iter,
    'optimizer': OPTIMIZER,
    'loss': type(mse).__name__,
    'input depth': input_depth,
    'input type': INPUT,
    'train_input': train_input
}
log_config.update(**freq_dict)
filename = os.path.basename(img_path).split('.')[0]
run = wandb.init(project="Fourier features DIP",
                 entity="impliciteam",
                 tags=['random_{}'.format(INPUT), 'depth:{}'.format(input_depth), filename],
                 name='{}_depth_{}_{}'.format(filename, input_depth, 'random_{}'.format(INPUT)),
                 job_type='train',
                 group='Inpainting',
                 mode='online',
                 save_code=True,
                 config=log_config,
                 notes='Input type {}, depth {}'.format(INPUT, input_depth))

# wandb.run.log_code(".")


if train_input:
    net_input_saved = net_input
else:
    net_input_saved = net_input.detach().clone()

noise = torch.rand_like(net_input) if INPUT == 'infer_freqs' else net_input.detach().clone()
# if INPUT == 'fourier':
#     indices = sample_indices(input_depth, net_input_saved)

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

# if INPUT == 'fourier':
#     net_input = net_input_saved[:, indices, :, :]
if INPUT == 'infer_freqs':
    net_input = generate_fourier_feature_maps(net_input_saved, (img_pil.size[1], img_pil.size[0]), dtype)
else:
    net_input = net_input_saved

out_np = torch_to_np(net(net_input))
log_images(np.array([np.clip(out_np, 0, 1), img_np]), num_iter, task='Inpainting')
plot_image_grid([out_np, img_np], factor=5)
plt.plot(psnr_gt_list)
plt.title('max: {}\nlast: {}'.format(max(psnr_gt_list), psnr_gt_list[-1]))
plt.show()
