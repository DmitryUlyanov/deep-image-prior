import shutil
from PIL import Image
from utils.inpainting_utils import get_random_img_mask, gen_text_mask
import glob
import tqdm
import os
import numpy as np


SOURCE_DATASET = './data/denoising_dataset/'
TARGET_DATASET = ['./data/inpainting_scribble_dataset/', './data/inpainting_text_dataset'][1]
os.makedirs(TARGET_DATASET, exist_ok=True)
source_imgs = sorted(glob.glob(SOURCE_DATASET + '/*.png'))

for src_img_path in tqdm.tqdm(source_imgs):
    src_img_name = os.path.basename(src_img_path)
    dst_img_path = os.path.join(TARGET_DATASET, src_img_name)
    shutil.copy(src_img_path, dst_img_path)
    img_pil = Image.open(src_img_path)
    img_np = np.array(img_pil)
    # curr_mask = get_random_img_mask(img_np)
    curr_mask = gen_text_mask(img_np)
    curr_mask_pil = Image.fromarray((curr_mask).astype(np.uint8))
    curr_mask_pil.save(os.path.join(TARGET_DATASET, 'mask_{}'.format(src_img_name)))


