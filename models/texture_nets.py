import torch
import torch.nn as nn
from .common import * 


normalization = nn.BatchNorm2d


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero'):
    if pad == 'zero':
        return nn.Conv2d(in_f, out_f, kernel_size, stride, padding=(kernel_size - 1) / 2, bias=bias)
    elif pad == 'reflection':
        layers = [nn.ReflectionPad2d((kernel_size - 1) / 2),
                  nn.Conv2d(in_f, out_f, kernel_size, stride, padding=0, bias=bias)]
        return nn.Sequential(*layers)

def get_texture_nets(inp=3, ratios = [32, 16, 8, 4, 2, 1], fill_noise=False, pad='zero', need_sigmoid=False, conv_num=8, upsample_mode='nearest'):


    for i in range(len(ratios)):
        j = i + 1

        seq = nn.Sequential()

        tmp =  nn.AvgPool2d(ratios[i], ratios[i])

        seq.add(tmp)
        if fill_noise:
            seq.add(GenNoise(inp))

        seq.add(conv(inp, conv_num, 3, pad=pad))
        seq.add(normalization(conv_num))
        seq.add(act())

        seq.add(conv(conv_num, conv_num, 3, pad=pad))
        seq.add(normalization(conv_num))
        seq.add(act())

        seq.add(conv(conv_num, conv_num, 1, pad=pad))
        seq.add(normalization(conv_num))
        seq.add(act())

        if i == 0:
            seq.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            cur = seq
        else:

            cur_temp = cur

            cur = nn.Sequential()

            # Batch norm before merging 
            seq.add(normalization(conv_num))
            cur_temp.add(normalization(conv_num * (j - 1)))

            cur.add(Concat(1, cur_temp, seq))

            cur.add(conv(conv_num * j, conv_num * j, 3, pad=pad))
            cur.add(normalization(conv_num * j))
            cur.add(act())

            cur.add(conv(conv_num * j, conv_num * j, 3, pad=pad))
            cur.add(normalization(conv_num * j))
            cur.add(act())

            cur.add(conv(conv_num * j, conv_num * j, 1, pad=pad))
            cur.add(normalization(conv_num * j))
            cur.add(act())

            if i == len(ratios) - 1: 
                cur.add(conv(conv_num * j, 3, 1, pad=pad))
            else:
                cur.add(nn.Upsample(scale_factor=2, mode=upsample_mode)) 
            
    model = cur
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
