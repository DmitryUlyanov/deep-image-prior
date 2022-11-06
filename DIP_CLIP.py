import sys
sys.path.append('./deep-image-prior-hqskipnet')
from models import *
from utils.sr_utils import *
import clip
import time
import numpy as np
import torch
import torch.optim
from IPython import display
import cv2
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import kornia.augmentation as K
from einops import rearrange
from madgrad import MADGRAD
import imageio
import random
import math

device = torch.device('cuda')

# clip_model = clip.load('ViT-B/16', device=device)[0]
# clip_model = clip.load('ViT-L/14', device=device)[0]
clip_model = clip.load('RN50x64', device=device)[0]
clip_model = clip_model.eval().requires_grad_(False)
#clip_size = 224
clip_size = 448
clip_normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.augs = T.Compose([
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(degrees=15, translate=0.1, p=0.8, padding_mode='border', resample='bilinear'),
            K.RandomPerspective(0.4, p=0.7, resample='bilinear'),
            K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7),
            K.RandomGrayscale(p=0.15),
        ])

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        if sideY != sideX:
            input = K.RandomAffine(degrees=0, shear=10, p=0.5)(input)

        max_size = min(sideX, sideY)
        cutouts = []
        for cn in range(self.cutn):
            if cn > self.cutn - self.cutn//4:
                cutout = input
            else:
                size = int(max_size * torch.zeros(1,).normal_(mean=.8, std=.3).clip(float(self.cut_size/max_size), 1.))
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        cutouts = torch.cat(cutouts)
        cutouts = self.augs(cutouts)
        return cutouts


def get_meshgrid(spatial_size):
    # X, Y = np.meshgrid(np.arange(1-spatial_size[1], spatial_size[1], 2) / float(spatial_size[1] - 1),
    #                    np.arange(1-spatial_size[0], spatial_size[0], 2) / float(spatial_size[0] - 1))
    X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                       np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
    meshgrid = np.concatenate([X[None, :], Y[None, :]])
    return meshgrid


def generate_fourier_feature_maps(net_input, spatial_size, dtype=torch.float32, only_cosine=False):
    meshgrid_np = get_meshgrid(spatial_size)
    meshgrid = torch.from_numpy(meshgrid_np).permute(1, 2, 0).unsqueeze(0).type(dtype)
    vp = net_input * torch.unsqueeze(meshgrid, -1)
    if only_cosine:
        vp_cat = torch.cat((torch.cos(vp),), dim=-1)
    else:
        vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    return vp_cat.flatten(-2, -1).permute(0, 3, 1, 2)


freqs = 2 ** torch.linspace(0., 7, steps=8)


def get_hq_skip_mlp_net(input_depth, pad='reflection', upsample_mode='cubic', n_channels=3, act_fun='LeakyReLU', skip_n33d=192, skip_n33u=192, skip_n11=4, num_scales=6, downsample_mode='cubic', decorr_rgb=True, offset_groups=4, offset_type='1x1'):
    """Constructs and returns a skip network with higher quality default settings, including
    deformable convolutions (can be slow, disable with offset_groups=0). Further
    improvements can be seen by setting offset_type to 'full', but then you may have to
    reduce the learning rate of the offset layers to ~1/10 of the rest of the layers. See
    the get_offset_params() and get_non_offset_params() functions to construct the
    parameter groups."""
    net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                        num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                        num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11,
                                        upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                        need_sigmoid=True, need_bias=True, decorr_rgb=decorr_rgb, pad=pad, act_fun=act_fun,
                                        offset_groups=offset_groups, offset_type=offset_type,
               filter_size_down=1, filter_size_up=1, filter_skip_size=1)
    return net


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def optimize_network(num_iterations, optimizer_type, lr):
    global itt
    itt = 0

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    make_cutouts = MakeCutouts(clip_size, cutn)

    # Initialize DIP skip network
    input_depth = 32
    # use Katherine Crowson's skip net
    from models import get_hq_skip_net
    net = get_hq_skip_mlp_net(input_depth).to(device)
    # net = get_hq_skip_net(input_depth).to(device)
    # net = get_net(
    #    input_depth, 'skip',
    #    pad='reflection',
    #    skip_n33d=128, skip_n33u=128,
    #    skip_n11=4, num_scales=7,
    #   upsample_mode='bilinear',
    # ).to(device)

    # Initialize input noise
    # net_input = torch.zeros([1, input_depth, sideY, sideX], device=device).normal_().div(10).detach()

    net_input = generate_fourier_feature_maps(freqs, (sideY, sideX), only_cosine=False).to(device)
    # Encode text prompt with CLIP
    target_embed = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr)
    elif optimizer_type == 'MADGRAD':
        optimizer = MADGRAD(net.parameters(), lr, weight_decay=0.01, momentum=0.9)

    try:
        for _ in range(num_iterations):
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                out = net(net_input).float()
            cutouts = make_cutouts(out)
            image_embeds = clip_model.encode_image(clip_normalize(cutouts))
            loss = spherical_dist_loss(image_embeds, target_embed).mean()

            loss.backward()
            optimizer.step()

            itt += 1

            if itt % display_rate == 0 or save_progress_video:
                with torch.inference_mode():
                    image = TF.to_pil_image(out[0].clamp(0, 1))
                    if itt % display_rate == 0:
                        display.clear_output(wait=True)
                        display.display(image)
                        # added to save image at each display rate - PF
                        image.save(f'dip_{timestring}_{itt}.png', quality=100)
                        if display_augs:
                            aug_grid = torchvision.utils.make_grid(cutouts, nrow=math.ceil(math.sqrt(cutn)))
                    if save_progress_video and itt > 15:
                        video_writer.append_data(np.asarray(image))

            if anneal_lr:
                optimizer.param_groups[0]['lr'] = max(0.00001, .99 * optimizer.param_groups[0]['lr'])

            print(f'Iteration {itt} of {num_iterations}')

    except KeyboardInterrupt:
        pass
    finally:
        return TF.to_pil_image(net(net_input)[0])


seed = random.randint(0, 2**32)
opt_type = 'MADGRAD' # Adam, MADGRAD
lr = 0.0025 # learning rate
anneal_lr = True # True == lower the learning rate over time

# sideX, sideY = 256, 256 # Resolution
# sideX, sizeY = 384, 384
# sideX, sizeY = 512, 256
# sideX, sizeY = 512, 384
sideX, sideY = 512,512
num_iterations = 700 # More can be better, but there are diminishing returns
# cutn of 10 fits in a V100 -PF
cutn = 10 # Number of crops of image shown to CLIP, this can affect quality

# prompt = 'a moody painting of a lonely duckling'
# prompt below from @RiversHaveWings
# prompt = 'the clockwork angel of [water/fire/earth/air] by Gerardo Dottori'
# prompt = 'a cute dog, sitting on the grass, fantasy art drawn by disney concept artists, golden colour, concept art, character concepts, digital painting, mystery, adventure'
prompt = 'a beautiful epic fantasy painting of [the wind/the soul/a skull/the sun]'
# prompt = "the oracle at Delphi by Gerardo Dottori"


display_rate = 50 # How often the output is displayed and saved. -PF
# If you grab a P100 GPU or better, you'll likely want to set this further apart, like >=20.
# On T4 and K80, the process is slower so you might want to set a faster display_rate (lower number, towards 1-5).

save_progress_video = False
display_augs = False # Display grid of augmented image, for debugging

# number of images you want to display and save - PF
num_images = 2
for i in range(0,num_images):
  timestring = time.strftime('%Y%m%d%H%M%S')
  if save_progress_video:
      video_writer = imageio.get_writer(f'dip_{timestring}.mp4', mode='I', fps=30, codec='libx264', quality=7, pixelformat='yuv420p')

  # Begin optimization / generation
  out = optimize_network(num_iterations, opt_type, lr)
  # Save final frame and video to a file
  out.save(f'pip_inp_16_{timestring}.png', quality=100)
  if save_progress_video:
     video_writer.close()
