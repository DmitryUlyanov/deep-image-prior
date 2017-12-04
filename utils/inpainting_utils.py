import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
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
