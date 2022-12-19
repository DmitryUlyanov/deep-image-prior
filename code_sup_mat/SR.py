from __future__ import print_function
from models import *
import random
import argparse
import os

from models.downsampler import Downsampler
from utils.sr_utils import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

parser = argparse.ArgumentParser()
parser.add_argument('--input_img_path', default='./data/F16_GT.png')
args = parser.parse_args()

# Fix seeds
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

imsize = -1
factor = 4
enforse_div32 = 'CROP'
PLOT = True
show_every = 100

path_to_image = args.input_img_path

# Starts here
imgs = load_LR_HR_imgs_sr_div_64(path_to_image, imsize, factor, enforse_div32)

imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(imgs['LR_pil'], imgs['HR_pil'])
output_depth = imgs['LR_np'].shape[0]
if PLOT:
    plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np']], 4,12);
    print('PSNR bicubic: %.4f   PSNR nearest: %.4f' % (
        compare_psnr(imgs['HR_np'], imgs['bicubic_np']),
        compare_psnr(imgs['HR_np'], imgs['nearest_np'])))

INPUT = 'infer_freqs'
pad = 'reflection'
if INPUT == 'infer_freqs':
    OPT_OVER = 'net,input'
else:
    OPT_OVER = 'net'

KERNEL_TYPE = 'lanczos2'
train_input = True if OPT_OVER != 'net' else False
LR = 0.01
tv_weight = 0.0
OPTIMIZER = 'adam'
freq_dict = {
    'method': 'log',
    'cosine_only': False,
    'n_freqs': 8,
    'base': 2 ** (8 / (8 - 1))
}

if factor == 4:
    num_iter = 2000
    reg_noise_std = 0.03
elif factor == 8:
    num_iter = 4001
    reg_noise_std = 0.05
else:
    assert False, 'We did not experiment with other factors'

input_depth = freq_dict['n_freqs'] * 4
net_input = get_input(input_depth, INPUT, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0]),
                      freq_dict=freq_dict).type(dtype)

print('Input is {}, Depth = {}'.format(INPUT, input_depth))

NET_TYPE = 'skip'  # UNet, ResNet
net = get_net(input_depth, 'skip', pad, n_channels=output_depth,
              skip_n33d=128,
              skip_n33u=128,
              skip_n11=4,
              num_scales=5,
              upsample_mode='bilinear').type(dtype)

# Losses
mse = torch.nn.MSELoss().type(dtype)

img_LR_var = np_to_torch(imgs['LR_np']).type(dtype)

downsampler = Downsampler(n_planes=output_depth,
                          factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)


def closure():
    global i, net_input

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
        net_input = generate_fourier_feature_maps(net_input_, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0]),
                                                  dtype,
                                                  freq_dict['cosine_only'])
    else:
        net_input = net_input_saved

    out_HR = net(net_input)
    out_LR = downsampler(out_HR)

    total_loss = mse(out_LR, img_LR_var)

    if tv_weight > 0:
        total_loss += tv_weight * tv_loss(out_HR)

    total_loss.backward()

    # Log
    psnr_LR = compare_psnr(imgs['LR_np'], torch_to_np(out_LR))
    psnr_HR = compare_psnr(imgs['HR_np'], torch_to_np(out_HR))

    # History
    psnr_history.append([psnr_LR, psnr_HR])

    if PLOT and i % show_every == 0:
        print('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f' % (i, psnr_LR, psnr_HR))

    i += 1

    return total_loss


psnr_history = []
if train_input:
    net_input_saved = net_input
else:
    net_input_saved = net_input.detach().clone()

noise = torch.rand_like(net_input) if INPUT == 'infer_freqs' else net_input.detach().clone()

i = 0

s = sum([np.prod(list(p.size())) for p in net.parameters()])
print('Number of params: %d' % s)
print(net)

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)
out_HR_np = np.clip(torch_to_np(net(net_input)), 0, 1)
result_deep_prior = put_in_center(out_HR_np, imgs['orig_np'].shape[1:])

plot_image_grid([imgs['HR_np'],
                 imgs['bicubic_np'],
                 out_HR_np], factor=4, nrow=1)

filename = os.path.basename(args.input_img_path).split('.')[0]
plt.imsave('{}_sr_learned_ff.png'.format(filename), out_HR_np.transpose(1, 2, 0))
plt.show()