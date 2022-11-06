import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.freq_utils import *


def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def get_params(opt_over, net, net_input, downsampler=None, input_encoder=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            if net_input.is_leaf:
                net_input.requires_grad = True
                params += [net_input]
            else:
                params += [x for x in input_encoder.parameters()]
        else:
            assert False, 'what is it?'

    return params


def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()


def plot_image_grid(images_np, nrow=8, factor=1, interpolation='lanczos'):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)

    plt.figure(figsize=(len(images_np) + factor, 12 + factor))

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    plt.show()

    return grid


def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img


def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size. 

    Args: 
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_meshgrid(spatial_size):
    # X, Y = np.meshgrid(np.arange(1-spatial_size[1], spatial_size[1], 2) / float(spatial_size[1] - 1),
    #                    np.arange(1-spatial_size[0], spatial_size[0], 2) / float(spatial_size[0] - 1))
    X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                       np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
    meshgrid = np.concatenate([X[None, :], Y[None, :]])
    return meshgrid


def get_input(input_depth, method, spatial_size, noise_type='u', var=1. / 10, freq_dict=None, img=None):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        meshgrid = get_meshgrid(spatial_size)
        net_input = np_to_torch(meshgrid)
    elif method == 'fourier':
        if freq_dict['method'] == 'linear':
            freqs = torch.linspace(0, freq_dict['max'], freq_dict['n_freqs'])
            net_input = generate_fourier_feature_maps(freqs, spatial_size, only_cosine=freq_dict['cosine_only'])
        elif freq_dict['method'] == 'random':
            meshgrid_np = get_meshgrid(spatial_size)
            meshgrid = torch.from_numpy(meshgrid_np).permute(1, 2, 0).unsqueeze(0)
            freqs = positional_encoding(v=meshgrid,
                                            m=40,
                                            sigma=10,
                                            sample_depth=input_depth // 4,
                                            scale_factor=4)
            net_input = generate_fourier_feature_maps(freqs, spatial_size, only_cosine=freq_dict['cosine_only'])
        elif freq_dict['method'] == 'log':
            freqs = freq_dict['base'] ** torch.linspace(0., freq_dict['n_freqs'] - 1, steps=freq_dict['n_freqs'])
            net_input = generate_fourier_feature_maps(freqs, spatial_size, only_cosine=freq_dict['cosine_only'])

    elif method == 'infer_freqs':
        meshgrid_np = get_meshgrid(spatial_size)
        meshgrid = torch.from_numpy(meshgrid_np).permute(1, 2, 0).unsqueeze(0)
        if freq_dict['method'] == 'linear':
            net_input = torch.linspace(1, freq_dict['base'] ** (freq_dict['n_freqs'] - 1), freq_dict['n_freqs'])
        elif freq_dict['method'] == 'random':
            net_input = positional_encoding(v=meshgrid,
                                            m=40,
                                            sigma=15,
                                            sample_depth=input_depth // 4)
        elif freq_dict['method'] == 'log':
            net_input = freq_dict['base'] ** torch.linspace(0., freq_dict['n_freqs'] - 1, steps=freq_dict['n_freqs'])
        elif freq_dict['method'] == 'learn2':
            net_input = meshgrid.float()
        elif freq_dict['method'] == 'wt':
            analyze_image(img.cpu().numpy(), size=64)
            # net_input = generate_fourier_feature_maps(freqs, spatial_size, only_cosine=freq_dict['cosine_only'])
            # if freq_dict['cosine_only']:
            #     assert input_depth == 2 * list(net_input.shape)[0]
            # else:
            #     assert input_depth == 4 * list(net_input.shape)[0]

    else:
        assert False

    return net_input


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')

        def closure2():
            optimizer.zero_grad()
            return closure()

        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        # optimizer = torch.optim.Adam([
        #     {'params': parameters[:-1]},
        #     {'params': parameters[-1], 'lr': 100*LR}], lr=LR)
        for j in tqdm.tqdm(range(num_iter)):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires=10, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 2,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def generate_fourier_feature_maps(net_input, spatial_size, dtype=torch.float32, only_cosine=False):
    meshgrid_np = get_meshgrid(spatial_size)
    meshgrid = torch.from_numpy(meshgrid_np).permute(1, 2, 0).unsqueeze(0).type(dtype)
    vp = net_input * torch.unsqueeze(meshgrid, -1)
    if only_cosine:
        vp_cat = torch.cat((torch.cos(vp),), dim=-1)
    else:
        vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    return vp_cat.flatten(-2, -1).permute(0, 3, 1, 2)


def positional_encoding(m, sigma, v, sample_depth=0, scale_factor=1):
    j = torch.arange(m, device=v.device)
    if sample_depth > 0:
        indices = torch.multinomial(torch.arange(0, j.shape[0], dtype=torch.float),
                                    sample_depth, replacement=False)
        coeffs = 2 * np.pi * sigma ** (j[indices] / m)
    else:
        coeffs = 2 * np.pi * sigma ** (j / m)

    coeffs = coeffs * scale_factor

    return coeffs


def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0  # to [16/255, 235/255]
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0  # to [16/255, 240/255]
    return im_ycbcr


def compare_psnr_y(x, y):
    return compare_psnr(rgb2ycbcr(x.transpose(1, 2, 0))[:, :, 0], rgb2ycbcr(y.transpose(1, 2, 0))[:, :, 0])


def np_cvt_color(img_np):
    """
    Convert image from BGR/RGB to RGb/BGR
    From B x C x W x H  to B x C x W x H
    """
    if len(img_np) == 4:
        return [img[::-1] for img in img_np]
    else:
        return img_np[::-1]

# https://github.com/willGuimont/learnable_fourier_positional_encoding/blob/master/learnable_fourier_pos_encoding.py
class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, C: int, M: tuple, F_dim: int, H_dim: int, D: int, gamma: float):
        """
        Learnable Fourier Features from https://arxiv.org/pdf/2106.02795.pdf (Algorithm 1)
        Implementation of Algorithm 1: Compute the Fourier feature positional encoding of a multi-dimensional position
        Computes the positional encoding of a tensor of shape [N, G, M]
        :param G: positional groups (positions in different groups are independent)
        :param M: each point has a M-dimensional positional values
        :param F_dim: depth of the Fourier feature dimension
        :param H_dim: hidden layer dimension
        :param D: positional encoding dimension
        :param gamma: parameter to initialize Wr
        """
        super().__init__()
        self.C = C
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma

        # Projection matrix on learned lines (used in eq. 2)
        self.Wr = nn.Linear(self.C, self.F_dim // 2, bias=False)
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            # nn.Linear(self.H_dim, self.D // self.C)
            nn.Linear(self.H_dim, self.D)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)
        # self.Wr.weight.data += (torch.pi / 2)

    def forward(self, x):
        """
        Produce positional encodings from x
        :param x: tensor of shape [N, G, M] that represents N positions where each position is in the shape of [G, M],
                  where G is the positional group and each group has M-dimensional positional values.
                  Positions in different positional groups are independent
        :return: positional encoding for X
        """
        B, H, W, C = x.shape
        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x.view(-1, C))
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        # Step 2. Compute projected Fourier features (eq. 6)
        Y = self.mlp(F)
        # Step 3. Reshape to x's shape
        PEx = Y.reshape((B, *self.M, self.D)).permute(0, 3, 1, 2)
        return PEx


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, C: int, M: tuple, F_dim: int, H_dim: int, D: int, gamma: float):
        """
        Learnable Fourier Features from https://arxiv.org/pdf/2106.02795.pdf (Algorithm 1)
        Implementation of Algorithm 1: Compute the Fourier feature positional encoding of a multi-dimensional position
        Computes the positional encoding of a tensor of shape [N, G, M]
        :param G: positional groups (positions in different groups are independent)
        :param M: each point has a M-dimensional positional values
        :param F_dim: depth of the Fourier feature dimension
        :param H_dim: hidden layer dimension
        :param D: positional encoding dimension
        :param gamma: parameter to initialize Wr
        """
        super().__init__()
        self.C = C
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma

        # Projection matrix on learned lines (used in eq. 2)
        self.Wr = nn.Linear(self.C, self.F_dim // 2, bias=False)
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            # nn.Linear(self.H_dim, self.D // self.C)
            nn.Linear(self.H_dim, self.D)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)
        # self.Wr.weight.data += (torch.pi / 2)

    def forward(self, x):
        """
        Produce positional encodings from x
        :param x: tensor of shape [N, G, M] that represents N positions where each position is in the shape of [G, M],
                  where G is the positional group and each group has M-dimensional positional values.
                  Positions in different positional groups are independent
        :return: positional encoding for X
        """
        B, H, W, C = x.shape
        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x.view(-1, C))
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        # Step 2. Compute projected Fourier features (eq. 6)
        Y = self.mlp(F)
        # Step 3. Reshape to x's shape
        PEx = Y.reshape((B, *self.M, self.D)).permute(0, 3, 1, 2)
        return PEx

# G = 2
# M = (512, 512)
# x = torch.from_numpy(get_meshgrid((512, 512))).unsqueeze(0).permute(0, 2, 3, 1).float()
# enc = LearnableFourierPositionalEncoding(G, M, 256, 128, 32, 10)
# pex = enc(x)
# print(pex.shape)


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
