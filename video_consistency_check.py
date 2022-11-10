import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models.video import r3d_18
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from math import exp
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class r2p1d18_loss(nn.Module):
    def __init__(self, requires_grad=False, loss_func=torch.nn.SmoothL1Loss(), compute_single_loss=True):
        super().__init__()
        a = r3d_18(pretrained=True)
        # vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stem = a.stem
        self.slice1 = a.layer1
        self.slice2 = a.layer2
        self.slice3 = a.layer3
        self.slice4 = a.layer4

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.loss_func = loss_func
        self.compute_single_loss = compute_single_loss  # False for vodeometric evaluate

    def forward(self, X1, X2):
        out = []

        for X in [X1, X2]:
            feat_list = []
            # print('passinslices')
            X = self.normalize_batch(X)
            X = self.reshape_batch(X)
            h = self.stem(X)
            feat_list.append(h)

            h = self.slice1(h)
            feat_list.append(h)

            h = self.slice2(h)
            feat_list.append(h)

            if self.compute_single_loss:
                out.append(feat_list)
                continue
            h = self.slice3(h)
            feat_list.append(h)
            h = self.slice4(h)
            feat_list.append(h)
            out.append(feat_list)
        losses = []
        for i in range(len(feat_list)):
            loss = self.loss_func(out[0][i], out[1][i])
            losses.append(loss)
            pass
            # vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
            # out.append(vgg_outputs(h_relu1_2,0,0,0))# h_relu2_2, h_relu3_3, h_relu4_3))
        losses = torch.stack(losses)
        if self.compute_single_loss:
            losses = torch.mean(losses)
        # https://github.com/pytorch/examples/blob/d91adc972cef0083231d22bcc75b7aaa30961863/fast_neural_style/neural_style/vgg.py
        return losses

    def normalize_batch(self, batch):
        # normalize using imagenet mean and std
        # mean = batch.new_tensor([0.43216, 0.394666, 0.37645]).view(-1, 1, 1)
        # std = batch.new_tensor([0.22803, 0.22145, 0.216989]).view(-1, 1, 1)
        # batch = batch.div_(255.0)
        T = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        return T(batch)

    def reshape_batch(self, batch):
        # batch : (NS)CHW
        NS, C, H, W = batch.shape
        if not batch.is_contiguous():
            batch = batch.contiguous()
        batch = batch.view(NS // 7, 7, C, H, W)  # NSCHW
        return batch.permute(0, 2, 1, 3, 4)  # NCSHW


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim3D(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


def avg_psnr(gt, pred):
    from skimage.metrics import peak_signal_noise_ratio
    return peak_signal_noise_ratio(gt, pred)


def main():
    from utils.video_utils import VideoDataset

    dataset = {
        'blackswan': {
            'gt': './data/eval_vid/clean_videos/blackswan.avi',
            'temp-pip': './data/eval_vid/denoised_videos/pip/blackswan_pip.mp4',
            'frame-by-frame': './data/eval_vid/denoised_videos/frame_by_frame/blackswan.mp4',
            '3d-dip': ''
        },
        'rollerblade': {
            'gt': './data/eval_vid/clean_videos/rollerblade.avi',
            'temp-pip': './data/eval_vid/denoised_videos/pip/rollerblade.mp4',
            'frame-by-frame': './data/eval_vid/denoised_videos/frame_by_frame/rollerblade.mp4',
            '3d-dip': './data/eval_vid/denoised_videos/3d_dip/rollerblade_3d_dip.mp4'
        },
        'tennis(40 frames)': {
            'gt': './data/eval_vid/clean_videos/tennis.avi',
            'temp-pip': './data/eval_vid/denoised_videos/pip/tennis_40_frames.mp4',
            'frame-by-frame': './data/eval_vid/denoised_videos/frame_by_frame/tennis.mp4',
            '3d-dip': './data/eval_vid/denoised_videos/3d_dip/tennis_3d_dip_12.mp4'
        },
        'judo': {
            'gt': './data/eval_vid/clean_videos/judo.mp4',
            'temp-pip': './data/eval_vid/denoised_videos/pip/judo.mp4',
            'frame-by-frame': './data/eval_vid/denoised_videos/frame_by_frame/judo.mp4',
            '3d-dip': ''
        },
    }
    chosen_video = [dataset['rollerblade'], dataset['tennis(40 frames)'], dataset['blackswan'], dataset['judo']][3]
    vid_gt = VideoDataset(chosen_video['gt'],
                          input_type='noise',
                          num_freqs=8,
                          task='denoising',
                          crop_shape=None,
                          batch_size=4,
                          arch_mode='3d',
                          mode='cont')

    vid_denoised = VideoDataset(chosen_video['temp-pip'],
                                input_type='noise',
                                num_freqs=8,
                                task='denoising',
                                crop_shape=None,
                                batch_size=4,
                                arch_mode='2d',
                                mode='cont')

    vid_denoised_frame_by_frame = VideoDataset(chosen_video['frame-by-frame'],
                                               input_type='noise',
                                               num_freqs=8,
                                               task='denoising',
                                               crop_shape=None,
                                               batch_size=4,
                                               arch_mode='2d',
                                               mode='cont')

    # vid_denoised_3d_dip = VideoDataset(chosen_video['3d-dip'],
    #                                    input_type='noise',
    #                                    num_freqs=8,
    #                                    task='denoising',
    #                                    crop_shape=None,
    #                                    batch_size=4,
    #                                    arch_mode='2d',
    #                                    mode='cont')

    gt = vid_gt.get_all_gt().cuda()
    denoised = vid_denoised.get_all_gt().cuda()
    denoised_f_b_f = vid_denoised_frame_by_frame.get_all_gt().cuda()
    # denoised_3d_dip = vid_denoised_3d_dip.get_all_gt().cuda()

    # remove edges  | rollerblade: 5 | tennis: 0 | Blackswan: 2 | Judo: 4
    remove_edges_start_index = 4
    if remove_edges_start_index > 0:
        gt = gt[:-remove_edges_start_index]
        denoised_f_b_f = denoised_f_b_f[:-remove_edges_start_index]
        # denoised_3d_dip = denoised_3d_dip[:-remove_edges_start_index]

    denoised = denoised[:-(remove_edges_start_index + 1)]

    ssim_loss = SSIM3D(window_size=11)
    print('3D-SSIM')
    print('temp-pip: {:.4f}'.format(ssim_loss(gt.permute(1, 0, 2, 3).unsqueeze(0),
                                              denoised.permute(1, 0, 2, 3).unsqueeze(0))))
    print('frame-by-frame: {:.4f}'.format(ssim_loss(gt.permute(1, 0, 2, 3).unsqueeze(0),
                                                    denoised_f_b_f.permute(1, 0, 2, 3).unsqueeze(0))))
    # print('3d-dip: {:.4f}'.format(ssim_loss(gt.permute(1, 0, 2, 3).unsqueeze(0),
    #                                         denoised_3d_dip.permute(1, 0, 2, 3).unsqueeze(0))))

    print('Avg. PSNR')
    print('temp-pip: {:.4f}'.format(avg_psnr(gt.cpu().numpy(), denoised.cpu().numpy())))
    print('frame-by-frame: {:.4f}'.format(avg_psnr(gt.cpu().numpy(), denoised_f_b_f.cpu().numpy())))
    # print('3d-dip: {:.4f}'.format(avg_psnr(gt.cpu().numpy(), denoised_3d_dip.cpu().numpy())))


if __name__ == '__main__':
    main()
