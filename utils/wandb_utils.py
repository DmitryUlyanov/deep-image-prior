import wandb
import numpy as np
from utils.common_utils import np_cvt_color


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
