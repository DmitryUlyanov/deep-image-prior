import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from .matcher import Matcher
from collections import OrderedDict

from torchvision.models.vgg import model_urls
from torchvision.models import vgg19
from torch.autograd import Variable

from .vgg_modified import VGGModified

def get_pretrained_net(name):
    """Loads pretrained network"""
    if name == 'alexnet_caffe':
        if not os.path.exists('alexnet-torch_py3.pth'):
            print('Downloading AlexNet')
            os.system('wget -O alexnet-torch_py3.pth --no-check-certificate -nc https://box.skoltech.ru/index.php/s/77xSWvrDN0CiQtK/download')
        return torch.load('alexnet-torch_py3.pth')
    elif name == 'vgg19_caffe':
        if not os.path.exists('vgg19-caffe-py3.pth'):
            print('Downloading VGG-19')
            os.system('wget -O vgg19-caffe-py3.pth --no-check-certificate -nc https://box.skoltech.ru/index.php/s/HPcOFQTjXxbmp4X/download')
        
        vgg = get_vgg19_caffe()
        
        return vgg
    elif name == 'vgg16_caffe':
        if not os.path.exists('vgg16-caffe-py3.pth'):
            print('Downloading VGG-16')
            os.system('wget -O vgg16-caffe-py3.pth --no-check-certificate -nc https://box.skoltech.ru/index.php/s/TUZ62HnPKWdxyLr/download')
        
        vgg = get_vgg16_caffe()
        
        return vgg
    elif name == 'vgg19_pytorch_modified':
        # os.system('wget -O data/feature_inversion/vgg19-caffe.pth --no-check-certificate -nc https://www.dropbox.com/s/xlbdo688dy4keyk/vgg19-caffe.pth?dl=1')
        
        model = VGGModified(vgg19(pretrained=False), 0.2)
        model.load_state_dict(torch.load('vgg_pytorch_modified.pkl')['state_dict'])

        return model
    else:
        assert False


class PerceputalLoss(nn.modules.loss._Loss):
    """ 
        Assumes input image is in range [0,1] if `input_range` is 'sigmoid', [-1, 1] if 'tanh' 
    """
    def __init__(self, input_range='sigmoid', 
                       net_type = 'vgg_torch', 
                       input_preprocessing='corresponding', 
                       match=[{'layers':[11,20,29],'what':'features'}]):
        
        if input_range not in ['sigmoid', 'tanh']:
            assert False

        self.net = get_pretrained_net(net_type).cuda()

        self.matchers = [get_matcher(self.net, match_opts) for match_opts in match]

        preprocessing_correspondence = {
            'vgg19_torch': vgg_preprocess_caffe,
            'vgg16_torch': vgg_preprocess_caffe,
            'vgg19_pytorch': vgg_preprocess_pytorch,
            'vgg19_pytorch_modified': vgg_preprocess_pytorch,
        }

        if input_preprocessing == 'corresponding':
            self.preprocess_input = preprocessing_correspondence[net_type]
        else:
            self.preprocessing = preprocessing_correspondence[input_preprocessing]

    def preprocess_input(self, x):
        if self.input_range == 'tanh':
            x = (x + 1.) / 2.

        return self.preprocess(x)

    def __call__(self, x, y):

        # for 
        self.matcher_content.mode = 'store'
        self.net(self.preprocess_input(y));
        
        self.matcher_content.mode = 'match'
        self.net(self.preprocess_input(x));
        
        return sum([sum(matcher.losses.values()) for matcher in self.matchers])


def get_vgg19_caffe():
    model = vgg19()
    model.classifier = nn.Sequential(View(), *model.classifier._modules.values())
    vgg = model.features
    vgg_classifier = model.classifier

    names = ['conv1_1','relu1_1','conv1_2','relu1_2','pool1',
             'conv2_1','relu2_1','conv2_2','relu2_2','pool2',
             'conv3_1','relu3_1','conv3_2','relu3_2','conv3_3','relu3_3','conv3_4','relu3_4','pool3',
             'conv4_1','relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','conv4_4','relu4_4','pool4',
             'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','conv5_4','relu5_4','pool5',
             'torch_view','fc6','relu6','drop6','fc7','relu7','drop7','fc8']
    
    model = nn.Sequential()
    for n, m in zip(names, list(vgg) + list(vgg_classifier)):
        model.add_module(n, m)

    model.load_state_dict(torch.load('vgg19-caffe-py3.pth'))

    return model

def get_vgg16_caffe():
    vgg = torch.load('vgg16-caffe-py3.pth')

    names = ['conv1_1','relu1_1','conv1_2','relu1_2','pool1',
             'conv2_1','relu2_1','conv2_2','relu2_2','pool2',
             'conv3_1','relu3_1','conv3_2','relu3_2','conv3_3','relu3_3','pool3',
             'conv4_1','relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','pool4',
             'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','pool5',
             'torch_view','fc6','relu6','drop6','fc7','relu7','fc8']
    
    model = nn.Sequential()
    for n, m in zip(names, list(vgg)):
        model.add_module(n, m)

    # model.load_state_dict(torch.load('vgg19-caffe-py3.pth'))

    return model


class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1) 


def get_matcher(vgg, opt):
    # idxs = [int(x) for x in opt['layers'].split(',')]
    matcher = Matcher(opt['what'], 'mse', opt['map_idx'])

    def hook(module, input, output):
        matcher(module, output)

    for layer_name in opt['layers']:
        vgg._modules[layer_name].register_forward_hook(hook)

    return matcher


def get_vgg(cut_idx=-1, vgg_type='pytorch'):
    f = get_vanilla_vgg_features(cut_idx, vgg_type)

    keys = [x for x in cnn._modules.keys()]
    max_idx = max(keys.index(x) for x in opt_content['layers'].split(','))
    for k in keys[max_idx+1:]:
        cnn._modules.pop(k)

    return f

vgg_mean = torch.FloatTensor([103.939, 116.779, 123.680]).view(3, 1, 1)
def vgg_preprocess_caffe(var):
    (r, g, b) = torch.chunk(var, 3, dim=1)
    bgr = torch.cat((b, g, r), 1)
    out = bgr * 255 - torch.autograd.Variable(vgg_mean).type(var.type())
    return out



mean_pytorch = Variable(torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
std_pytorch =  Variable(torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1))

def vgg_preprocess_pytorch(var):
    return (var - mean_pytorch.type_as(var))/std_pytorch.type_as(var)



def get_preprocessor(imsize):
    def vgg_preprocess(tensor):
        (r, g, b) = torch.chunk(tensor, 3, dim=0)
        bgr = torch.cat((b, g, r), 0)
        out = bgr * 255 - vgg_mean.type(tensor.type()).expand_as(bgr)
        return out
    preprocess = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor(),
        transforms.Lambda(vgg_preprocess)
    ])

    return preprocess


def get_deprocessor():
    def vgg_deprocess(tensor):
        bgr = (tensor + vgg_mean.expand_as(tensor)) / 255.0
        (b, g, r) = torch.chunk(bgr, 3, dim=0)
        rgb = torch.cat((r, g, b), 0)
        return rgb
    deprocess = transforms.Compose([
        transforms.Lambda(vgg_deprocess),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        transforms.ToPILImage()
    ])
    return deprocess

