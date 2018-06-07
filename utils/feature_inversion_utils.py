import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from .matcher import Matcher
import os
from collections import OrderedDict

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(-1) 

def get_vanilla_vgg_features(cut_idx=-1):
    if not os.path.exists('vgg_features.pth'):
        os.system(
            'wget --no-check-certificate -N https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg19-d01eb7cb.pth')
        vgg_weights = torch.load('vgg19-d01eb7cb.pth')
        # fix compatibility issues
        map = {'classifier.6.weight':u'classifier.7.weight', 'classifier.6.bias':u'classifier.7.bias'}
        vgg_weights = OrderedDict([(map[k] if k in map else k,v) for k,v in vgg_weights.iteritems()])

        

        model = models.vgg19()
        model.classifier = nn.Sequential(View(), *model.classifier._modules.values())
        

        model.load_state_dict(vgg_weights)
        
        torch.save(model.features, 'vgg_features.pth')
        torch.save(model.classifier, 'vgg_classifier.pth')

    vgg = torch.load('vgg_features.pth')
    if cut_idx > 36:
        vgg_classifier = torch.load('vgg_classifier.pth')
        vgg = nn.Sequential(*(vgg._modules.values() + vgg_classifier._modules.values()))

    vgg.eval()

    return vgg


def get_matcher(net, opt):
    idxs = [x for x in opt['layers'].split(',')]
    matcher = Matcher(opt['what'])

    def hook(module, input, output):
        matcher(module, output)

    for i in idxs:
        net._modules[i].register_forward_hook(hook)

    return matcher



def get_vgg(cut_idx=-1):
    f = get_vanilla_vgg_features(cut_idx)

    if cut_idx > 0: 
        num_modules = len(f._modules)
        keys_to_delete = [f._modules.keys()[x] for x in range(cut_idx, num_modules)]
        for k in keys_to_delete:
            del f._modules[k]

    return f

def vgg_preprocess_var(var):
        (r, g, b) = torch.chunk(var, 3, dim=1)
        bgr = torch.cat((b, g, r), 1)
        out = bgr * 255 - torch.autograd.Variable(vgg_mean[None, ...]).type(var.type()).expand_as(bgr)
        return out

vgg_mean = torch.FloatTensor([103.939, 116.779, 123.680]).view(3, 1, 1)



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
