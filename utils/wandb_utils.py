import wandb
import numpy as np
from utils.common_utils import np_cvt_color


def log_images(array_of_imgs, iter, task, psnr=None):
    B, C, H, W = array_of_imgs.shape
    images_np = np.zeros((H, B * W, C), dtype=np.float32)
    for i in range(B):
        images_np[:, i * W:W * (i + 1), :] = np.transpose(array_of_imgs[i], (1, 2, 0))

    if psnr is not None:
        caption = 'Iteration #{}\n PSNR: {}'.format(iter, psnr)
    else:
        caption = 'Iteration #{}'.format(iter)
    wandb.log({task: wandb.Image(images_np, caption=caption)})


def log_input_images(degraded_img, gt_img):
    gt_img_np = gt_img.transpose(1, 2, 0)
    degraded_img_np = degraded_img.transpose(1, 2, 0)

    wandb.log({'Degraded Img': wandb.Image(degraded_img_np)})
    wandb.log({'GT Img': wandb.Image(gt_img_np)})


def log_input_video(gt_sequence, degraded_sequence):
    B, C, H, W = gt_sequence.shape
    wandb.log({'Degraded Video': [wandb.Image(np_cvt_color(degraded_sequence[frame_idx]).transpose(1, 2, 0),
                                              caption='Frame #{}'.format(frame_idx)) for frame_idx in range(B)]},
              commit=False)
    wandb.log({'Clean Video': [wandb.Image(np_cvt_color(gt_sequence[frame_idx]).transpose(1, 2, 0),
                                           caption='Frame #{}'.format(frame_idx)) for frame_idx in range(B)]},
              commit=False)


def log_inputs(inputs):
    inputs_ = inputs.squeeze(0).detach().cpu().numpy()
    C, H, W = inputs_.shape
    inputs_arr = np.zeros((C, H, W), np.float32)
    for channel_idx in range(C):
        inputs_arr[channel_idx] = inputs_[channel_idx, :, :]

    wandb.log({'Input': [wandb.Image(inputs_arr[ch_idx], caption='channel #{}'.format(ch_idx)) for ch_idx in range(C)]},
              commit=False)
