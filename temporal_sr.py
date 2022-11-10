from __future__ import print_function

import glob
import random

from models import *
from models.skip_3d import skip_3d, skip_3d_mlp, skip_3d_mlp_enc_inp
from utils.denoising_utils import *
from utils.wandb_utils import *
from utils.video_utils import VideoDataset, select_frames
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

INPUT = ['noise', 'fourier', 'meshgrid', 'infer_freqs'][args.input_index]
vid_dataset = VideoDataset(args.input_vid_path,
                           input_type=INPUT,
                           num_freqs=args.num_freqs,
                           task='temporal_sr',
                           crop_shape=None,
                           batch_size=8,
                           arch_mode='2d',
                           mode='cont',
                           train=True)

vid_dataset_eval = VideoDataset(args.input_vid_path,
                                input_type=INPUT,
                                num_freqs=args.num_freqs,
                                task='temporal_sr',
                                crop_shape=None,
                                batch_size=8,
                                mode='cont',
                                arch_mode='2d',
                                train=False)
pad = 'reflection'
if INPUT == 'infer_freqs':
    OPT_OVER = 'net,input'
else:
    OPT_OVER = 'net'

train_input = True if ',' in OPT_OVER else False
reg_noise_std = 0  # 1. / 30.  # set to 1./20. for sigma=50
LR = args.learning_rate

OPTIMIZER = 'adam'  # 'LBFGS'
exp_weight = 0.99
show_every = 3000
n_epochs = 25000
num_iter = 1
figsize = 4

if INPUT == 'noise':
    input_depth = vid_dataset.input_depth
    net = skip_3d(input_depth, 3,
                  num_channels_down=[16, 32, 64, 128, 128, 128],
                  num_channels_up=[16, 32, 64, 128, 128, 128],
                  num_channels_skip=[4, 4, 4, 4, 4, 4],
                  filter_size_up=(3, 3, 3),
                  filter_size_down=(3, 3, 3),
                  filter_size_skip=(1, 1, 1),
                  downsample_mode='stride',
                  need1x1_up=True, need_sigmoid=True, need_bias=True, pad='reflection',
                  act_fun='LeakyReLU').type(dtype)
else:
    input_depth = args.num_freqs * 6  # 4 * F for spatial encoding, 4 * F for temporal encoding
    # net = skip_3d_mlp(input_depth, 3,
    #                   num_channels_down=[256, 256, 256, 256, 256, 256],
    #                   num_channels_up=[256, 256, 256, 256, 256, 256],
    #                   num_channels_skip=[8, 8, 8, 8, 8, 8],
    #                   filter_size_up=(1, 1, 1),
    #                   filter_size_down=(1, 1, 1),
    #                   filter_size_skip=(1, 1, 1),
    #                   downsample_mode='stride',
    #                   need1x1_up=True, need_sigmoid=True, need_bias=True, pad='reflection',
    #                   act_fun='LeakyReLU').type(dtype)
    net = skip(input_depth, 3,
               num_channels_down=[256, 256, 256, 256, 256, 256],
               num_channels_up=[256, 256, 256, 256, 256, 256],
               num_channels_skip=[8, 8, 8, 8, 8, 8],
               filter_size_up=1,
               filter_size_down=1,
               filter_skip_size=1,
               upsample_mode='bilinear',
               downsample_mode='stride',
               need1x1_up=True, need_sigmoid=True, need_bias=True, pad='reflection',
               act_fun='LeakyReLU').type(dtype)

# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()])
print('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

last_net = None
psnr_lr_last = 0
psnr_hr_list = []
best_psnr_gt = -1.0
i = 0


def eval_video(val_dataset, model, epoch):
    spatial_size = vid_dataset.get_cropped_video_dims()
    img_for_video = np.zeros((val_dataset.n_frames, 3, *spatial_size), dtype=np.uint8)
    img_for_psnr = np.zeros((val_dataset.n_frames, 3, *spatial_size), dtype=np.float32)

    val_dataset.init_batch_list()
    with torch.no_grad():
        while True:
            batch_data = val_dataset.next_batch()
            if batch_data is None:
                break
            batch_idx = batch_data['batch_idx']
            batch_data = val_dataset.prepare_batch(batch_data)

            net_out = model(batch_data['input_batch'])
            out = net_out  # N x 3 x H x W
            out_np = out.detach().cpu().numpy()

            img_for_psnr[batch_data['cur_batch']] = out_np
            out_rgb = np.array([np_cvt_color(o) for o in out_np])
            img_for_video[batch_data['cur_batch']] = (out_rgb * 255).astype(np.uint8)

    ignore_start_ind = vid_dataset_eval.n_batches * vid_dataset_eval.batch_size
    psnr_whole_video = compare_psnr(val_dataset.get_all_gt(numpy=True)[:ignore_start_ind],
                                    img_for_psnr[:ignore_start_ind])
    wandb.log({'Checkpoint (FPS=10)'.format(epoch): wandb.Video(img_for_video, fps=10, format='mp4'),
               'Checkpoint (FPS=25)'.format(epoch): wandb.Video(img_for_video, fps=25, format='mp4'),
               'Video PSNR': psnr_whole_video},
              commit=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'temporal_sr_checkpoint_{}.pth'.format(epoch))


def train_batch(batch_data):
    global j

    net_input_saved = batch_data['input_batch']
    noise = net_input_saved.detach().clone()
    if INPUT == 'noise':
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        else:
            net_input = net_input_saved
    elif INPUT == 'fourier':
        net_input = net_input_saved

    net_out = net(net_input)
    out = net_out.squeeze(0)  # N x 3 x H x W
    # out_lr = select_frames(out)
    out_lr = out
    # out_hr = out
    total_loss = mse(out_lr, batch_data['img_noisy_batch'])
    total_loss.backward()

    # out_hr_np = out_hr.detach().cpu().numpy()
    out_lr_np = out_lr.detach().cpu().numpy()
    psnr_lr = compare_psnr(batch_data['img_noisy_batch'].cpu().numpy(), out_lr_np)
    # psnr_hr = compare_psnr(batch_data['gt_batch'].numpy(), out_hr_np)

    # wandb.log({'batch loss': total_loss.item(), 'psnr_lr': psnr_lr}, commit=True)
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
    return total_loss, psnr_lr


p = get_params(OPT_OVER, net, net_input=vid_dataset.input)
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
                       'PIP'],
                 name='{}_depth_{}_{}_factor_6_sequential'.format(filename, input_depth, '{}'.format(INPUT)),
                 job_type='sequential_{}_{}'.format(INPUT, LR),
                 group='Video - Temporal SR',
                 mode='online',
                 save_code=True,
                 config=log_config,
                 notes=''
                 )

log_input_video(vid_dataset.get_all_gt(numpy=True),
                vid_dataset.get_all_degraded(numpy=True))

wandb.run.log_code(".", exclude_fn=lambda path: path.find('venv') != -1)
print(net)
n_batches = vid_dataset.n_batches

for epoch in tqdm.tqdm(range(n_epochs), desc='Epoch'):
    batch_cnt = 0
    running_psnr = 0.
    running_loss = 0.
    vid_dataset.init_batch_list()
    for batch_cnt in tqdm.tqdm(range(n_batches), desc="Batch", position=1, leave=False):
        batch_data = vid_dataset.next_batch()
        # batch_data = vid_dataset.sample_next_batch()
        # batch_idx = batch_data['batch_idx']
        batch_data = vid_dataset.prepare_batch(batch_data)
    # batch_data = vid_dataset.sample_next_batch()
    # batch_data = vid_dataset.prepare_batch(batch_data)
        for j in range(num_iter):
            optimizer.zero_grad()
            loss, psnr_lr = train_batch(batch_data)
            running_loss += loss.item()
            running_psnr += psnr_lr
            optimizer.step()

    # Log metrics for each epoch
    wandb.log({'epoch loss': running_loss / n_batches, 'epoch psnr_lr': running_psnr / n_batches}, commit=True)
    # log_images(np.array([np_cvt_color(o) for o in out_sequence]), epoch, 'Video-TemporalSR',
    #            commit=False)

    # Infer video:
    if epoch % show_every == 0:
        eval_video(vid_dataset_eval, net, epoch)

# Infer video at the end:
eval_video(vid_dataset_eval, net, epoch)
