from __future__ import print_function

import glob
import random

from models import *
from models.skip_3d import skip_3d, skip_3d_mlp
from utils.denoising_utils import *
from utils.wandb_utils import *
from utils.video_utils import VideoDataset
from utils.common_utils import np_cvt_color
import torch.optim
import matplotlib.pyplot as plt

import os
import wandb
import argparse
import numpy as np
import cv2
# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

# Fix seeds
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0')
parser.add_argument('--input_vid_path', default='', type=str, required=True)
parser.add_argument('--input_index', default=0, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--num_freqs', default=8, type=int)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
imsize = -1
PLOT = True
sigma = 25
sigma_ = sigma / 255.

INPUT = ['noise', 'fourier', 'meshgrid', 'infer_freqs'][args.input_index]
vid_dataset = VideoDataset(args.input_vid_path,
                           input_type=INPUT,
                           num_freqs=args.num_freqs,
                           resize_shape=(192, 384))

pad = 'reflection'
if INPUT == 'infer_freqs':
    OPT_OVER = 'net,input'
else:
    OPT_OVER = 'net'

train_input = True if ',' in OPT_OVER else False
reg_noise_std = 0  # 1. / 30.  # set to 1./20. for sigma=50
LR = args.learning_rate

OPTIMIZER = 'adam'  # 'LBFGS'
show_every = 100
exp_weight = 0.99

num_iter = 100
figsize = 4

if INPUT == 'noise':
    input_depth = 32
    net = skip_3d(input_depth, 3,
                  num_channels_down=[128, 128, 128, 128, 128, 128],
                  num_channels_up=[128, 128, 128, 128, 128, 128],
                  num_channels_skip=[4, 4, 4, 4, 4, 4],
                  filter_size_up=(3, 3, 3),
                  filter_size_down=(3, 3, 3),
                  filter_size_skip=(1, 1, 1),
                  downsample_mode='stride',
                  need1x1_up=True, need_sigmoid=True, need_bias=True, pad='reflection',
                  act_fun='LeakyReLU').type(dtype)
else:
    input_depth = args.num_freqs * 6  # 4 * F for spatial encoding, 2 * F for temporal encoding
    net = skip_3d_mlp(input_depth, 3,
                      num_channels_down=[128, 128, 128, 128, 128, 128],
                      num_channels_up=[128, 128, 128, 128, 128, 128],
                      num_channels_skip=[4, 4, 4, 4, 4, 4],
                      filter_size_up=(1, 1, 1),
                      filter_size_down=(1, 1, 1),
                      filter_size_skip=(1, 1, 1),
                      downsample_mode='stride',
                      need1x1_up=True, need_sigmoid=True, need_bias=True, pad='reflection',
                      act_fun='LeakyReLU').type(dtype)

# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()])
print('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

# if train_input:
#     net_input_saved = net_input
# else:
#     net_input_saved = net_input.detach().clone()
#
# noise = torch.rand_like(net_input) if INPUT == 'infer_freqs' else net_input.detach().clone()

last_net = None
psrn_noisy_last = 0
psnr_gt_list = []
best_psnr_gt = -1.0
best_iter = 0
best_img = None
i = 0
spatial_size = vid_dataset.get_video_dims()


def train_batch(batch_data):
    global j
    best_loss_recon_image = 1e9
    best_iter = 0

    net_input_saved = batch_data['input_batch']
    if INPUT == 'noise':
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        else:
            net_input = net_input_saved
    elif INPUT == 'fourier':
        net_input = net_input_saved

    # elif INPUT == 'infer_freqs':
    #     if reg_noise_std > 0:
    #         net_input_ = net_input_saved + (noise.normal_() * reg_noise_std)
    #     else:
    #         net_input_ = net_input_saved
    #
    #     net_input = generate_fourier_feature_maps(net_input_, *spatial_size, dtype)

    net_out = net(net_input)
    out = net_out.squeeze(0).transpose(0, 1)  # N x 3 x H x W
    total_loss = mse(out, batch_data['img_noisy_batch'])
    total_loss.backward()

    out_np = out.detach().cpu().numpy()
    psrn_noisy = compare_psnr(batch_data['img_noisy_batch'].cpu().numpy(), out_np)
    psrn_gt = compare_psnr(batch_data['gt_batch'].numpy(), out_np)

    # print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f' % (
    #     j, total_loss.item(), psrn_noisy, psrn_gt))
    # psnr_gt_list.append(psrn_gt)
    # wandb.log({'psnr_gt': psrn_gt, 'psnr_noisy': psrn_noisy}, commit=False)
    # if psrn_gt > best_psnr_gt:
    #     best_psnr_gt = psrn_gt
    #     best_img = np.copy(out_np)
    #     best_iter = i
    # Backtracking
    # if i % show_every:
    #     if psrn_noisy - psrn_noisy_last < -2:
    #         print('Falling back to previous checkpoint.')
    #
    #         for new_param, net_param in zip(last_net, net.parameters()):
    #             net_param.data.copy_(new_param.cuda())
    #
    #         return total_loss * 0
    #     else:
    #         last_net = [x.detach().cpu() for x in net.parameters()]
    #         psrn_noisy_last = psrn_noisy

    # wandb.log({'training loss': total_loss.item()}, commit=True)
    return total_loss, psrn_gt, out_np


p = get_params(OPT_OVER, net, None)
optimizer = torch.optim.Adam(p, lr=LR)
log_config = {
    "learning_rate": LR,
    "iteration per batch": num_iter,
    'optimizer': OPTIMIZER,
    'loss': type(mse).__name__,
    'input depth': input_depth,
    'input type': INPUT,
    'Train input': train_input,
    'Reg. Noise STD': reg_noise_std,
    'Sequence length': vid_dataset.batch_size,
    'Video length': vid_dataset.n_frames,
    '# of sequences': vid_dataset.n_batches
}
log_config.update(**vid_dataset.freq_dict)
filename = os.path.basename(args.input_vid_path).split('.')[0]
run = wandb.init(project="Fourier features DIP",
                 entity="impliciteam",
                 tags=['{}'.format(INPUT), 'depth:{}'.format(input_depth), filename, vid_dataset.freq_dict['method']],
                 name='{}_depth_{}_{}'.format(filename, input_depth, '{}'.format(INPUT)),
                 job_type='{}_{}'.format(INPUT, LR),
                 group='Denoising - Video',
                 mode='offline',
                 save_code=True,
                 config=log_config,
                 notes='Baseline'
                 )

log_input_video(vid_dataset.get_all_gt(numpy=True),
                vid_dataset.get_all_degraded(numpy=True))

wandb.run.log_code(".", exclude_fn=lambda path: path.find('venv') != -1)
print(net)

batch_cnt = 0
img_for_video = np.zeros((vid_dataset.n_frames, 3, *spatial_size), dtype=np.uint8)

while True:
    # Get batch data
    batch_data = vid_dataset.next_batch()
    if batch_data is None:
        break
    batch_idx = batch_data['batch_idx']
    batch_data = vid_dataset.prepare_batch(batch_data)
    print('{}/{} : Batch range: {}'.format(batch_cnt, vid_dataset.n_batches, batch_data['cur_batch']))
    for j in tqdm.tqdm(range(num_iter)):
        optimizer.zero_grad()
        loss, psnr, out_sequence = train_batch(batch_data)
        optimizer.step()

    # Log metrics per sequence
    wandb.log({'training loss': loss.item(), 'psnr': psnr}, commit=True)
    log_images(np.array([np_cvt_color(o) for o in out_sequence]), num_iter * (batch_cnt + 1), 'Video-Denoising')

    # Infer images at the end of the batch
    with torch.no_grad():
        net_out = net(batch_data['input_batch'])
        out = net_out.squeeze(0).transpose(0, 1).detach().cpu().numpy()  # N x 3 x H x W
        out_rgb = np.array([np_cvt_color(o) for o in out])
        img_for_video[batch_data['cur_batch']] = (out_rgb * 255).astype(np.uint8)

    batch_cnt += 1

wandb.log({'Final Sequence': wandb.Video(img_for_video)})


