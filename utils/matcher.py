import torch
import torch.nn as nn

class Matcher:
    def __init__(self, how='gram_matrix', loss='mse'):
        self.mode = 'store'
        self.stored = {}
        self.losses = {}

        if how in all_features.keys():
            self.get_statistics = all_features[how]
        else:
            assert False
        pass

        if loss in all_losses.keys():
            self.loss = all_losses[loss]
        else:
            assert False

    def __call__(self, module, features):
        statistics = self.get_statistics(features)

        self.statistics = statistics
        if self.mode == 'store':
            self.stored[module] = statistics.detach().clone()
        elif self.mode == 'match':
            self.losses[module] = self.loss(statistics, self.stored[module])

    def clean(self):
        self.losses = {}

def gram_matrix(x):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def features(x):
    return x


all_features = {
    'gram_matrix': gram_matrix,
    'features': features,
}

all_losses = {
    'mse': nn.MSELoss(),
    'smoothL1': nn.SmoothL1Loss(),
    'L1': nn.L1Loss(),
}
