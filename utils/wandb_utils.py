import wandb
import numpy as np
from utils.common_utils import np_cvt_color
import matplotlib.pyplot as plt
from torch.fft import fft2, fftshift


def log_images(array_of_imgs, iter, task, psnr=None, commit=True):
    B, C, H, W = array_of_imgs.shape
    images_np = np.zeros((H, B * W, C), dtype=np.float32)
    for i in range(B):
        images_np[:, i * W:W * (i + 1), :] = np.transpose(array_of_imgs[i], (1, 2, 0))

    if psnr is not None:
        caption = 'Iteration #{}\n PSNR: {}'.format(iter, psnr)
    else:
        caption = 'Iteration #{}'.format(iter)
    wandb.log({task: wandb.Image(images_np, caption=caption)}, commit=commit)


def log_input_images(degraded_img, gt_img):
    gt_img_np = gt_img.transpose(1, 2, 0)
    degraded_img_np = degraded_img.transpose(1, 2, 0)

    wandb.log({'Degraded Img': wandb.Image(degraded_img_np)})
    wandb.log({'GT Img': wandb.Image(gt_img_np)})


def log_input_video(gt_sequence, degraded_sequence):
    B, C, H, W = degraded_sequence.shape
    imgs_for_degraded = np.array(
        [np_cvt_color((degraded_sequence[frame_idx]) * 255).astype(np.uint8)
        for frame_idx in range(B)])
    B, C, H, W = gt_sequence.shape
    imgs_for_gt = np.array(
        [np_cvt_color((gt_sequence[frame_idx]) * 255).astype(np.uint8) for frame_idx in range(B)])

    wandb.log({'Degraded Video (FPS=10)': wandb.Video(imgs_for_degraded, fps=10, format='mp4'),
              'Degraded Video (FPS=25)': wandb.Video(imgs_for_degraded, fps=25, format='mp4')},
              commit=False)
    wandb.log({'Clean Video (FPS=10)': wandb.Video(imgs_for_gt, fps=10, format='mp4'),
               'Clean Video (FPS=25)': wandb.Video(imgs_for_gt, fps=25, format='mp4')},
              commit=False)


def log_inputs(inputs):
    inputs_ = inputs.squeeze(0).detach().cpu().numpy()
    C, H, W = inputs_.shape
    inputs_arr = np.zeros((C, H, W), np.float32)
    for channel_idx in range(C):
        inputs_arr[channel_idx] = inputs_[channel_idx, :, :]

    wandb.log({'Input': [wandb.Image(inputs_arr[ch_idx], caption='channel #{}'.format(ch_idx)) for ch_idx in range(C)]},
              commit=False)


f1_ref = 0
f2_ref = 0
f3_ref = 0


def visualize_fourier(img, iter, is_gt=False):
    global f1_ref, f2_ref, f3_ref
    if is_gt:
        iter='gt'

    H, W = img.shape[-2:]
    f = fft2(img)
    f = fftshift(f)
    mag = f.abs()
    fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    mag_norm = (mag/mag.max())[0]
    ax.matshow(mag_norm, cmap='jet')
    title = 'GT - 2D FT' if is_gt else '2D - FT'
    plt.title(title)
    plt.savefig('./plots/2d_ft_{}.png'.format(iter))
    figx = plt.figure()
    arr = mag_norm.squeeze(-1).numpy()[:, H // 2]
    plt.bar(x=range(-H // 2, H // 2), height=arr)
    inds_f3 = np.argsort(arr)[::-1][1:3]  # taking 2 peaks after bias
    inds_f2 = np.argsort(arr)[::-1][3:5]  # taking 2 peaks after bias
    inds_f1 = np.argsort(arr)[::-1][5:7]  # taking 2 peaks after bias
    if is_gt:
        f3_ref = arr[inds_f3]
        f2_ref = arr[inds_f2]
        f1_ref = arr[inds_f1]
    else:
        wandb.log({'Diff from f#3 peaks': np.sum(np.abs(f3_ref - arr[inds_f3]))})
        wandb.log({'Diff from f#2 peaks': np.sum(np.abs(f2_ref - arr[inds_f2]))})
        wandb.log({'Diff from f#1 peaks': np.sum(np.abs(f1_ref - arr[inds_f1]))})

    plt.title('FT - X axis')
    plt.xlabel('Frequency')
    plt.xlabel('Amplitude')
    plt.xlim([-100, 100])
    title = 'GT - X - FT' if is_gt else 'X - FT'
    plt.title(title)
    plt.savefig('./plots/ft_x_{}.png'.format(iter))
    plt.close('all')


