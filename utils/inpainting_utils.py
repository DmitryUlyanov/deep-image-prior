import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from random import randint
import cv2
from .common_utils import *


def get_text_mask(for_image, sz=20):
    font_fname = '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf'
    font_size = sz
    font = ImageFont.truetype(font_fname, font_size)

    img_mask = Image.fromarray(np.array(for_image)*0+255)
    draw = ImageDraw.Draw(img_mask)
    draw.text((128, 128), "hello world", font=font, fill='rgb(0, 0, 0)')

    return img_mask


def get_bernoulli_mask(for_image, zero_fraction=0.95):
    img_mask_np=(np.random.random_sample(size=pil_to_np(for_image).shape) > zero_fraction).astype(int)
    img_mask = np_to_pil(img_mask_np)
    
    return img_mask


def get_random_img_mask(for_image):
    H, W, C = for_image.shape
    img = np.zeros((H, W, C), np.uint8)

    # Set size scale
    size = int((W + H) * 0.1)
    if W < 64 or H < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    # Draw random lines
    x1, x2 = randint(W//4, 3*(W//4)), randint(W//3, 3*(W//4))
    y1, y2 = randint(H//4, 3*(H//4)), randint(H//4, 3*(H//4))
    thickness = randint(3, size)
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness)

    # Draw random circles
    x1, y1 = randint(W//4, 3*(W//4)), randint(H//4, 3*(H//4))
    radius = randint(3, size)
    cv2.circle(img, (x1, y1), radius, (255, 255, 255), -1)

    # Draw random ellipses
    # x1, y1 = randint(1, W), randint(1, H)
    # s1, s2 = randint(1, W), randint(1, H)
    # a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
    # thickness = randint(3, size)
    # cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (255, 255, 255), thickness)

    return 255 - img


def gen_text_mask(for_image):
    H, W, C = for_image.shape
    text_mask_pil = Image.open('./data/inpainting_text_dataset/text_mask2.png').resize((W, H), Image.BICUBIC)
    text_mask_np = np.array(text_mask_pil)
    text_mask_binary = (text_mask_np > 128).astype(np.float) * 255
    return text_mask_binary


