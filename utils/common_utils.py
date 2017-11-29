import torch
import torch.nn as nn
import torchvision
import sys

from torch.autograd import Variable
import numpy as np
from PIL import Image
import PIL
import numpy as np

import matplotlib.pyplot as plt

def crop_image(img, d=32):
    '''
        Make dimensions divisible by `d`
    '''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            (img.size[0] - new_size[0])/2, 
            (img.size[1] - new_size[1])/2,
            (img.size[0] + new_size[0])/2,
            (img.size[1] + new_size[1])/2,
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def get_params(opt_over, net, net_input, downsampler=None):
    '''
        comma separated list
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

def tv_loss(x):
    h = torch.sum(torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]))
    v = torch.sum(torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]))
    return h + v    


def get_image_grid(images_np, nrow=8):
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=1, interpolation=None):
    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np)+factor,12+factor))
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1,2,0), interpolation=interpolation)
    plt.show()
    
    return grid

def load(path):
    img = Image.open(path)#.convert('RGB')
    return img

def get_image(path, imsize):
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np



def fill_noise(x, noise_type):
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = Variable(torch.zeros(shape))
        
        fill_noise(net_input.data, noise_type)
        net_input.data *= var 
    elif method == 'noise+meshgrid':
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        meshgrid -= (0 if noise_type=='u' else 0.5)
        mg =  np_to_var(meshgrid)

        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = Variable(torch.zeros(shape))
        
        fill_noise(net_input.data, noise_type)
        net_input.data *= var 

        net_input.data[0,:2] = mg.data
            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_var(meshgrid)
    else:
        assert False
        
    return net_input

def pil_to_np(img_PIL):
    '''
        from W x H x C [0...255] tot C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32)/255.

def np_to_pil(img_np): 
    '''
        from C x W x H [0..1] to  W x H x C [0...255]
    '''

    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1,2,0)

    return Image.fromarray(ar)

def np_to_tensor(img_np):
    '''
        from numpy.array to torch.Tensor
    '''
    return torch.from_numpy(img_np)

def np_to_var(img_np, dtype = torch.cuda.FloatTensor):
    '''
        from numpy.array to torch.Variable
    '''
    return Variable(np_to_tensor(img_np)[None, :])

def var_to_np(img_var):
    '''
        from torch.Variable to np.array
    '''
    return img_var.data.cpu().numpy()[0]


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass  


def optimize(optimizer_type, parameters, closure, LR, num_iter, downsampler=None):
    """
        Runs optimization loop.
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False