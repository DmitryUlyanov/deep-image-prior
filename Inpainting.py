from __future__ import print_function

from utils.inpainting_utils import *
from utils.wandb_utils import *
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
from models.mlp import MLP
from models.simple_fcn import FCN
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
parser.add_argument('--input_index', default=0, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--num_freqs', default=8, type=int)
parser.add_argument('--dataset_index', default=0, type=int)
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

INPUT = ['noise', 'fourier', 'meshgrid', 'infer_freqs'][args.input_index]
if args.index == -1:
    # fnames = sorted(glob.glob('data/inpainitng_scribble_set5/*.*'))
    fnames = sorted(glob.glob('data/inpainting_scribble_dataset/*.*'))
    # fnames = sorted(glob.glob('data/inpainting_text_dataset/*.*'))
    mask_paths = [f for f in fnames if 'mask' in f]
    img_paths = [f for f in fnames if 'mask' not in f]

    img_path = img_paths[args.dataset_index]
    mask_path = mask_paths[args.dataset_index]
else:
    img_path = img_dict[args.index]['img_path']
    mask_path = img_dict[args.index]['mask_path']

NET_TYPE = 'skip_depth6'  # 'MLP'  # 'FCN'  # one of skip_depth4|skip_depth2|UNET|ResNet


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
        'n_freqs': args.num_freqs,
        'base': 2 ** (8 / (args.num_freqs-1))
    }
pad = 'reflection'  # 'zero'
OPTIMIZER = 'adam'

if 'vase.png' in img_path:
    # INPUT = 'meshgrid' # 'infer_freqs'
    input_depth = freq_dict['n_freqs'] * 4
    LR = 0.01
    num_iter = 8001
    param_noise = False
    show_every = 50
    figsize = 5
    reg_noise_std = 0.03

    net = skip(input_depth, img_np.shape[0],
               # num_channels_down=[128] * 5,
               # num_channels_up=[128] * 5,
               # num_channels_skip=[0] * 5,
               # upsample_mode='nearest', filter_skip_size=1, filter_size_up=3, filter_size_down=3,
               # need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
               num_channels_down=[128] * 5,
               num_channels_up=[128] * 5,
               num_channels_skip=[4] * 5,
               filter_size_up=1, filter_size_down=1, filter_skip_size=1,
               upsample_mode='bilinear',  # downsample_mode='avg',
               need1x1_up=True,
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

    # net = MLP(input_depth, out_dim=3, hidden_list=[256, 256, 256, 256]).type(dtype)

elif ('kate.png' in img_path) or ('peppers.png' in img_path):
    # Same params and net as in super-resolution and denoising
    # INPUT = 'infer_freqs'  # 'fourier'  # 'noise'
    input_depth = args.num_freqs * 4
    LR = 0.01
    num_iter = 6001
    param_noise = False
    show_every = 50
    figsize = 5
    reg_noise_std = 0.03

    net = skip(input_depth, img_np.shape[0],
               num_channels_down=[128] * 5,
               num_channels_up=[128] * 5,
               num_channels_skip=[4] * 5,
               filter_size_up=1, filter_size_down=1,
               upsample_mode='bilinear', filter_skip_size=1,
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

    # net = MLP(input_depth, out_dim=3, hidden_list=[256, 256, 256, 256]).type(dtype)
    # net = FCN(input_depth, out_dim=3, hidden_list=[256, 256, 256, 256]).type(dtype)

elif 'library.png' in img_path:
    # INPUT = 'noise'  # 'infer_freqs'
    input_depth = args.num_freqs * 4

    num_iter = 8001
    show_every = 50
    figsize = 8
    reg_noise_std = 0.00
    param_noise = True

    if 'skip' in NET_TYPE:

        depth = int(NET_TYPE[-1])
        net = skip(input_depth, img_np.shape[0],
                   # num_channels_down=[16, 32, 64, 128, 128, 128][:depth],
                   # num_channels_up=[16, 32, 64, 128, 128, 128][:depth],
                   # num_channels_skip=[0, 0, 0, 0, 0, 0][:depth],
                   # filter_size_up=3, filter_size_down=5, filter_skip_size=1,
                   # upsample_mode='nearest',  # downsample_mode='avg',
                   # need1x1_up=False,
                   # need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
                   num_channels_down=[128] * 5,
                   num_channels_up=[128] * 5,
                   num_channels_skip=[4] * 5,
                   filter_size_up=1, filter_size_down=1, filter_skip_size=1,
                   upsample_mode='bilinear',  #downsample_mode='avg',
                   need1x1_up=True,
                   need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

        LR = args.learning_rate

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

    elif NET_TYPE == 'MLP':
        net = MLP(input_depth, out_dim=3, hidden_list=[256, 256, 256, 256]).type(dtype)
        LR = args.learning_rate

    elif NET_TYPE == 'FCN':
        net = FCN(input_depth, out_dim=3, hidden_list=[256, 256, 256, 256]).type(dtype)
        LR = args.learning_rate

    else:
        assert False
else:
    input_depth = 1 if INPUT == 'noise' else args.num_freqs * 4
    LR = args.learning_rate
    num_iter = 3000
    param_noise = False
    show_every = 50
    figsize = 5
    reg_noise_std = 0.03

    # net = skip(input_depth, img_np.shape[0],
    #            num_channels_down=[128] * 6,
    #            num_channels_up=[128] * 6,
    #            num_channels_skip=[4] * 6,
    #            # num_channels_down=[16, 32, 64, 128, 128, 128][:6],
    #            # num_channels_up=[16, 32, 64, 128, 128, 128][:6],
    #            # num_channels_skip=[4, 4, 4, 4, 4, 4][:6],
    #            filter_size_up=1, filter_size_down=1,
    #            upsample_mode='bilinear', filter_skip_size=1,
    #            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

    net = skip(input_depth, img_np.shape[0],
               num_channels_down=[16, 32, 64, 128, 128, 128][:6],
               num_channels_up=[16, 32, 64, 128, 128, 128][:6],
               num_channels_skip=[0, 0, 0, 0, 0, 0][:6],
               filter_size_up=3, filter_size_down=5, filter_skip_size=1,
               upsample_mode='nearest',  # downsample_mode='avg',
               need1x1_up=False,
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

net = net.type(dtype)
net_input = get_input(input_depth, INPUT, (img_pil.size[1], img_pil.size[0]), freq_dict=freq_dict).type(dtype)
noise = net_input.detach().clone()

if INPUT == 'infer_freqs':
    OPT_OVER = 'net,input'
else:
    OPT_OVER = 'net'

train_input = True if ',' in OPT_OVER else False

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_var = np_to_torch(img_np).type(dtype)
mask_var = np_to_torch(img_mask_np).type(dtype)
psnr_gt_list = []
psnr_mask_list = []
psnr_masked_last = 0.0
last_net = None
best_psnr_gt = -1.0
best_img = None
best_iter = 0
i = 0


def closure():
    global i, last_net, psnr_masked_last, best_psnr_gt, best_img, best_iter

    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std() / 50

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
        net_input = generate_fourier_feature_maps(net_input_, (img_pil.size[1], img_pil.size[0]), dtype,
                                                  only_cosine=freq_dict['cosine_only'])
    else:
        net_input = net_input_saved

    out = net(net_input)

    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()

    if PLOT and i % show_every == 0:
        out_np = out.detach().cpu().numpy()[0]
        psnr_gt = compare_psnr(img_np, out_np)
        psnr_masked = compare_psnr(img_np * img_mask_np, out_np)
        psnr_gt_list.append(psnr_gt)
        psnr_mask_list.append(psnr_masked)

        print('Iteration %05d    Loss %f    psnr_gt %f   psnr_masked %f' % (i, total_loss.item(), psnr_gt, psnr_masked))
        wandb.log({'psnr_gt': psnr_gt, 'psnr_noisy': psnr_masked}, commit=False)
        if psnr_gt > best_psnr_gt:
            best_psnr_gt = psnr_gt
            best_img = np.copy(out_np)
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
                 tags=['{}'.format(INPUT), 'depth:{}'.format(input_depth), filename, freq_dict['method']],
                 name='{}_depth_{}_{}'.format(filename, input_depth, '{}'.format(INPUT)),
                 job_type='Set5_{}_{}'.format(LR, INPUT),
                 group='Inpainting',
                 mode='online',
                 save_code=True,
                 config=log_config,
                 notes='')

wandb.run.log_code(".", exclude_fn=lambda path: path.find('venv') != -1)
log_input_images(img_np * img_mask_np, img_np)
# Compute number of parameters
s = sum(np.prod(list(p.size())) for p in net.parameters())
print('Number of params: %d' % s)
print(net)
if train_input:
    net_input_saved = net_input
else:
    net_input_saved = net_input.detach().clone()

noise = torch.rand_like(net_input) if INPUT == 'infer_freqs' else net_input.detach().clone()

p = get_params(OPT_OVER, net, net_input)
if train_input:
    if INPUT == 'infer_freqs':
        net_input = generate_fourier_feature_maps(net_input_saved, (img_pil.size[1], img_pil.size[0]), dtype,
                                                  only_cosine=freq_dict['cosine_only'])
    else:
        log_inputs(net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)
if INPUT == 'infer_freqs':
    net_input = generate_fourier_feature_maps(net_input_saved, (img_pil.size[1], img_pil.size[0]), dtype,
                                              only_cosine=freq_dict['cosine_only'])
    if train_input:
        log_inputs(net_input)
else:
    net_input = net_input_saved

out_np = torch_to_np(net(net_input))
os.makedirs('./plots/inpainting/', exist_ok=True)
plt.imsave('./plots/inpainting/{}_{}_{}.png'.format(filename, INPUT, psnr_gt_list[-1]), np.clip(out_np.transpose(1, 2, 0), 0, 1))
# log_images(np.array([np.clip(out_np, 0, 1), img_np]), num_iter, task='Inpainting')
# log_images(np.array([np.clip(best_img, 0, 1), img_np]), best_iter, task='Best Image', psnr=best_psnr_gt)
log_images(np.array([np.clip(out_np, 0, 1)]), num_iter, task='Inpainting')
log_images(np.array([np.clip(best_img, 0, 1)]), best_iter, task='Best Image', psnr=best_psnr_gt)
plot_image_grid([out_np, img_np], factor=5)
plt.plot(psnr_gt_list)
plt.title('max: {}\nlast: {}'.format(max(psnr_gt_list), psnr_gt_list[-1]))
plt.show()
run.finish()
