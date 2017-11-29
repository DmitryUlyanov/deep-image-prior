import cv2
import os
from common_utils import *
import shutil
import tempfile
from contextlib import contextmanager


@contextmanager
def TemporaryDirectory():
    name = tempfile.mkdtemp()
    try:
        yield name
    finally:
        shutil.rmtree(name)


        
def get_noisy_image(img_np, sigma):
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np
    
def baseline_denoise(img_noisy_pil, method, sigma):
    if method == 'nlm':
        ar = np.array(img_noisy_pil)
        if len(ar.shape) == 3:
            out = Image.fromarray(cv2.fastNlMeansDenoisingColored(ar, None, 10, 10, 7, 21))
        else:
            out = Image.fromarray(cv2.fastNlMeansDenoising(ar[:, :, None], None, 10, 7, 21))
    elif method == 'bm3d':
        
        with TemporaryDirectory() as d:
            img_noisy_pil.save('%s/tmp.png' % d)
            
            noisy= "%s/noisy.png" % d
            basic= "%s/basic.png" % d
            denoised= "%s/denoised.png" % d
            diff= "%s/diff.png" % d
            bias= "%s/bias.png" % d
            diffbias = "%s/diffbias.png" % d
            
            os.system("./bm3d/BM3Ddenoising %s/tmp.png %d %s %s %s %s %s %s 1 bior 0 dct 1 opp" % (d, sigma,
                                                                                                      noisy,
                                                                                                      basic,
                                                                                                      denoised,
                                                                                                      diff,
                                                                                                      bias,
                                                                                                      diffbias))
            out = Image.open(denoised)
    else:
        assert False
        
    return out