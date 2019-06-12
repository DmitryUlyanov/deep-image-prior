# This script had been used to get the numbers in the paper
from utils.common_utils import get_image, plot_image_grid
import cv2
def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    return im_ycbcr

def compare_psnr_y(x, y):
    return compare_psnr(rgb2ycbcr(x.transpose(1,2,0))[:,:,0], rgb2ycbcr(y.transpose(1,2,0))[:,:,0])
    
from collections import defaultdict
datasets = { 
    'Set14':   ["baboon", "barbara", "bridge", "coastguard", "comic", "face", "flowers", "foreman", "lenna", "man", "monarch", "pepper", "ppt3", "zebra"],
#     'Set5':    ['baby', 'bird', 'butterfly', 'head', 'woman']
}
from glob import glob
# g = sorted(glob('../image_compare/data/sr/Set5/x4/*'))

from skimage.measure import compare_psnr
# our 
stats = {}
imsize  = -1
dct = defaultdict(lambda : 0)
for cur_dataset in datasets.keys():
    
    for method_name in postfixes:
        psnrs = []
        for name in datasets[cur_dataset]:
            img_HR = f'/home/dulyanov/dmitryulyanov.github.io/assets/deep-image-prior/SR/{cur_dataset}/x4/{name}_GT.png'
            ours   = f'/home/dulyanov/dmitryulyanov.github.io/assets/deep-image-prior/SR/{cur_dataset}/x4/{name}_deep_prior.png'
            method = f'/home/dulyanov/dmitryulyanov.github.io/assets/deep-image-prior/SR/{cur_dataset}/x4/{name}_{method_name}.png'

            gt_pil,     gt      = get_image(img_HR, imsize)
            ours_pil,   ours    = get_image(ours, imsize)
            method_pil, methods = get_image(method, imsize)
            
            if methods.shape[0] == 1:
                methods = np.concatenate([methods, methods, methods], 0)
    
            q1 = ours[:3].sum(0)
            t1 = np.where(q1.sum(0) > 0)[0]
            t2 = np.where(q1.sum(1) > 0)[0]



            psnr = compare_psnr_y(gt     [:3,t2[0] + 4:t2[-1]-4,t1[0] + 4:t1[-1] - 4], 
                                  methods[:3,t2[0] + 4:t2[-1]-4,t1[0] + 4:t1[-1] - 4])

    #         psnr = compare_psnr(gt  [:3], 
    #                             ours[:3])

            psnrs.append(psnr)

            print(name, psnr)

        
        header = f'\small{{{method_name}}} & ' + ' & '.join([f'${x:.4}$' for x in psnrs])
        
        stats[method_name] = [header, np.mean(psnrs)]
        
        print (header)
        
    names = datasets[cur_dataset]
    header = ' & ' + ' & '.join([f'\small{{{x.title()}}}' for x in names])    
