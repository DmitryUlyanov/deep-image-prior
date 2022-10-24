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
import tqdm
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
                           resize_shape=(192, 384),
                           batch_size=8)


vid_dataset_eval = VideoDataset(args.input_vid_path,
                                input_type=INPUT,
                                num_freqs=args.num_freqs,
                                resize_shape=(192, 384),
                                batch_size=8,
                                mode='cont')
pad = 'reflection'
if INPUT == 'infer_freqs':
    OPT_OVER = 'net,input'
else:
    OPT_OVER = 'net'

train_input = True if ',' in OPT_OVER else False
reg_noise_std = 0  # 1. / 30.  # set to 1./20. for sigma=50
LR = args.learning_rate

OPTIMIZER = 'adam'  # 'LBFGS'
show_every = 300
exp_weight = 0.99
n_epochs = 1500
num_iter = 1
figsize = 4

if INPUT == 'noise':
    input_depth = 1
    net = skip_3d(input_depth, 3,
                  num_channels_down=[16, 32, 64, 128, 128, 128],
                  num_channels_up=[16, 32, 64, 128, 128, 128],
                  num_channels_skip=[4, 4, 4, 4, 4, 4],
                  filter_size_up=(3, 3, 3),
                  filter_size_down=(3, 5, 5),
                  filter_size_skip=(1, 1, 1),
                  downsample_mode='stride',
                  need1x1_up=True, need_sigmoid=True, need_bias=True, pad='reflection',
                  act_fun='LeakyReLU').type(dtype)
else:
    input_depth = args.num_freqs * 8  # 4 * F for spatial encoding, 4 * F for temporal encoding
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


def eval_video(val_dataset, model, epoch):
    img_for_video = np.zeros((vid_dataset.n_frames, 3, *spatial_size), dtype=np.uint8)
    img_for_psnr = np.zeros((vid_dataset.n_frames, 3, *spatial_size), dtype=np.float32)
    border = val_dataset.get_batch_size() // 2

    with torch.no_grad():
        while True:
            batch_data = val_dataset.next_batch()
            if batch_data is None:
                break
            batch_idx = batch_data['batch_idx']
            batch_data = val_dataset.prepare_batch(batch_data)

            net_out = model(batch_data['input_batch'])
            out = net_out.squeeze(0).transpose(0, 1)  # N x 3 x H x W
            out_center = out[3, :, :, :]
            out_np = out_center.detach().cpu().numpy()

            img_for_psnr[batch_idx + 3] = out_np
            # out_rgb = np.array([np_cvt_color(o) for o in out_np])
            out_rgb = np_cvt_color(out_np)
            img_for_video[batch_idx + 3] = (out_rgb * 255).astype(np.uint8)

    psnr_whole_video = compare_psnr(val_dataset.get_all_gt(numpy=True)[border:-border],
                                    img_for_psnr[border:-border])
    wandb.log({'Checkpoint'.format(epoch): wandb.Video(img_for_video), 'Video PSNR': psnr_whole_video})


def train_batch(batch_data):
    global j

    net_input_saved = batch_data['input_batch']
    if INPUT == 'noise':
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        else:
            net_input = net_input_saved
    elif INPUT == 'fourier':
        net_input = net_input_saved

    net_out = net(net_input)
    out = net_out.squeeze(0).transpose(0, 1)  # N x 3 x H x W
    total_loss = mse(out, batch_data['img_noisy_batch'])
    total_loss.backward()

    out_np = out.detach().cpu().numpy()
    psrn_noisy = compare_psnr(batch_data['img_noisy_batch'].cpu().numpy(), out_np)
    psrn_gt = compare_psnr(batch_data['gt_batch'].numpy(), out_np)

    wandb.log({'batch loss': total_loss.item(), 'psnr_gt': psrn_gt, 'psnr_noisy': psrn_noisy}, commit=True)
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
    '# of sequences': vid_dataset.n_batches,
    'save every': show_every
}
log_config.update(**vid_dataset.freq_dict)
filename = os.path.basename(args.input_vid_path).split('.')[0]
run = wandb.init(project="Fourier features DIP",
                 entity="impliciteam",
                 tags=['{}'.format(INPUT), 'depth:{}'.format(input_depth), filename, vid_dataset.freq_dict['method'],
                       'temporal_stride'],
                 name='{}_depth_{}_{}'.format(filename, input_depth, '{}'.format(INPUT)),
                 job_type='{}_{}'.format(INPUT, LR),
                 group='Denoising - Video',
                 mode='online',
                 save_code=True,
                 config=log_config,
                 notes='Inferring center frame only | Temporal Stride = 2'
                 )

log_input_video(vid_dataset.get_all_gt(numpy=True),
                vid_dataset.get_all_degraded(numpy=True))

wandb.run.log_code(".", exclude_fn=lambda path: path.find('venv') != -1)
print(net)
n_batches = vid_dataset.n_batches

for epoch in tqdm.tqdm(range(n_epochs), desc='Epoch', position=0):
    batch_cnt = 0
    running_psnr = 0.
    running_loss = 0.
    vid_dataset.init_batch_list()
    for batch_cnt in tqdm.tqdm(range(n_batches), desc="Batch", position=1, leave=False):
        batch_data = vid_dataset.next_batch()
        batch_idx = batch_data['batch_idx']
        batch_data = vid_dataset.prepare_batch(batch_data)
        for j in range(num_iter):
            optimizer.zero_grad()
            loss, psnr, out_sequence = train_batch(batch_data)
            # if j == 0:
            #     wandb.log({'first iteration psnr': psnr}, commit=False)
            running_loss += loss.item()
            running_psnr += psnr
            optimizer.step()

    # Log metrics for each epoch
    wandb.log({'epoch loss': running_loss / n_batches, 'epoch psnr': running_psnr / n_batches}, commit=False)
    log_images(np.array([np_cvt_color(o) for o in out_sequence]), num_iter * (batch_cnt + 1), 'Video-Denoising',
               commit=False)

    # Infer video:
    if epoch % show_every == 0:
        eval_video(vid_dataset_eval, net, epoch)

# Infer video at the end:
eval_video(vid_dataset_eval, net, epoch)
