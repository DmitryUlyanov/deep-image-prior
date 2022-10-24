######################################################################
# Code modified from https://github.com/DmitryUlyanov/deep-image-prior
######################################################################

import torch
import torch.nn as nn
from .common import *


def skip_3d_mlp(
        num_input_channels=1, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=(1, 1, 1), filter_size_up=(1, 1, 1), filter_size_skip=(1, 1, 1),
        upsample_mode='trilinear', downsample_mode='stride',
        need_sigmoid=True, need_bias=True, need1x1_up=True,
        pad='zero', act_fun='LeakyReLU',
):
    """Assembles encoder-decoder with skip connections, using 3D convolutions.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max' (default: 'stride')
    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not isinstance(upsample_mode, list):
        upsample_mode = [upsample_mode] * n_scales

    if not isinstance(downsample_mode, list):
        downsample_mode = [downsample_mode] * n_scales

    if not isinstance(filter_size_down, list):
        filter_size_down = [filter_size_down] * n_scales

    if not isinstance(filter_size_up, list):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        bn_up = BatchNorm3D(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i]))
        model_tmp.add(bn_up)

        if num_channels_skip[i] != 0:
            conv_skip = conv3d(input_depth, num_channels_skip[i], filter_size_skip, bias=need_bias, pad=pad)
            bn_skip = BatchNorm3D(num_channels_skip[i])
            skip.add(conv_skip)
            skip.add(bn_skip)
            skip.add(act(act_fun))

        conv_down = conv3d(input_depth, num_channels_down[i], filter_size_down[i], stride=(2, 2, 2), bias=need_bias,
                           pad=pad, downsample_mode=downsample_mode[i])
        bn_down = BatchNorm3D(num_channels_down[i])
        deeper.add(conv_down)
        # deeper.add(PrintLayer())
        # deeper.add(AvgPool3D())
        deeper.add(bn_down)
        deeper.add(act(act_fun))

        conv_down = conv3d(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad)
        bn_down = BatchNorm3D(num_channels_down[i])
        deeper.add(conv_down)
        deeper.add(bn_down)
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        conv_up = conv3d(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], bias=need_bias, pad=pad)
        bn_up = BatchNorm3D(num_channels_up[i])
        model_tmp.add(conv_up)
        model_tmp.add(bn_up)
        model_tmp.add(act(act_fun))

        if need1x1_up:
            conv_up = conv3d(num_channels_up[i], num_channels_up[i], kernel_size=(1, 1, 1), bias=need_bias, pad=pad)
            bn_up = BatchNorm3D(num_channels_up[i])
            model_tmp.add(conv_up)
            model_tmp.add(bn_up)
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    conv_final = conv3d(num_channels_up[0], num_output_channels, kernel_size=(1, 1, 1), bias=need_bias, pad=pad)
    model.add(conv_final)
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


def skip_3d(
        num_input_channels=1, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=(3, 3, 3), filter_size_up=(3, 3, 3), filter_size_skip=(1, 1, 1),
        upsample_mode='nearest', downsample_mode='stride',
        need_sigmoid=True, need_bias=True, need1x1_up=True,
        pad='zero', act_fun='LeakyReLU',
):
    """Assembles encoder-decoder with skip connections, using 3D convolutions.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max' (default: 'stride')
    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not isinstance(upsample_mode, list):
        upsample_mode = [upsample_mode] * n_scales

    if not isinstance(downsample_mode, list):
        downsample_mode = [downsample_mode] * n_scales

    if not isinstance(filter_size_down, list):
        filter_size_down = [filter_size_down] * n_scales

    if not isinstance(filter_size_up, list):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        bn_up = BatchNorm3D(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i]))
        model_tmp.add(bn_up)

        if num_channels_skip[i] != 0:
            conv_skip = conv3d(input_depth, num_channels_skip[i], filter_size_skip, bias=need_bias, pad=pad)
            bn_skip = BatchNorm3D(num_channels_skip[i])
            skip.add(conv_skip)
            skip.add(bn_skip)
            skip.add(act(act_fun))

        conv_down = conv3d(input_depth, num_channels_down[i], filter_size_down[i], stride=(1, 2, 2), bias=need_bias,
                           pad=pad, downsample_mode=downsample_mode[i])
        bn_down = BatchNorm3D(num_channels_down[i])
        deeper.add(conv_down)
        deeper.add(bn_down)
        deeper.add(act(act_fun))

        conv_down = conv3d(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad)
        bn_down = BatchNorm3D(num_channels_down[i])
        deeper.add(conv_down)
        deeper.add(bn_down)
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(Upsample3D(scale_factor=2, mode=upsample_mode[i]))

        conv_up = conv3d(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], bias=need_bias, pad=pad)
        bn_up = BatchNorm3D(num_channels_up[i])
        model_tmp.add(conv_up)
        model_tmp.add(bn_up)
        model_tmp.add(act(act_fun))

        if need1x1_up:
            conv_up = conv3d(num_channels_up[i], num_channels_up[i], kernel_size=(1, 1, 1), bias=need_bias, pad=pad)
            bn_up = BatchNorm3D(num_channels_up[i])
            model_tmp.add(conv_up)
            model_tmp.add(bn_up)
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    conv_final = conv3d(num_channels_up[0], num_output_channels, kernel_size=(1, 1, 1), bias=need_bias, pad=pad)
    model.add(conv_final)
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


def conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=True, pad='zero',
           downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_channels, factor=stride, kernel_type=downsample_mode, phase=0.5,
                                      preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    pad_D = int((kernel_size[0] - 1) / 2)
    pad_HW = int((kernel_size[1] - 1) / 2)
    to_pad = (pad_D, pad_HW, pad_HW)
    if pad == 'reflection':
        padder = ReflectionPad3D(pad_D, pad_HW)
        to_pad = (0, 0, 0)

    convolver = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)


class BatchNorm3D(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm3D, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        assert (x.size(0) == 1)  # 1 x C x D x H x W
        y = x.squeeze(0).transpose(0, 1).contiguous()  # D x C x H x W
        y = self.bn(y)
        y = y.transpose(0, 1).unsqueeze(0)
        return y


class Upsample3D(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Upsample3D, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        assert (x.size(0) == 1)  # 1 x C x D x H x W
        y = x.squeeze(0).transpose(0, 1)  # D x C x H x W
        y = self.upsample(y)
        return y.transpose(0, 1).unsqueeze(0)


class ReflectionPad3D(nn.Module):
    def __init__(self, pad_D, pad_HW):
        super(ReflectionPad3D, self).__init__()
        self.padder_HW = nn.ReflectionPad2d(pad_HW)
        self.padder_D = nn.ReplicationPad3d((0, 0, 0, 0, pad_D, pad_D))

    def forward(self, x):
        assert (x.size(0) == 1)  # 1 x C x D x H x W
        y = x.squeeze(0).transpose(0, 1)  # D x C x H x W
        y = self.padder_HW(y)
        y = y.transpose(0, 1).unsqueeze(0)  # 1 x C x D x H x W
        return self.padder_D(y)
