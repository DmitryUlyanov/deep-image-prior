import torch
import torch.nn as nn


class Matcher:
    def __init__(self, how='gram_matrix', loss='mse', map_index=933):
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

        self.map_index = map_index
        self.method = 'match'


    def __call__(self, module, features):
        statistics = self.get_statistics(features)

        self.statistics = statistics
        if self.mode == 'store':
            self.stored[module] = statistics.detach()

        elif self.mode == 'match':
           
            if statistics.ndimension() == 2:

                if self.method == 'maximize':
                    self.losses[module] = - statistics[0, self.map_index]
                else:
                    self.losses[module] = torch.abs(300 - statistics[0, self.map_index]) 

            else:
                ws = self.window_size

                t = statistics.detach() * 0

                s_cc = statistics[:1, :, t.shape[2] // 2 - ws:t.shape[2] // 2 + ws, t.shape[3] // 2 - ws:t.shape[3] // 2 + ws] #* 1.0
                t_cc = t[:1, :, t.shape[2] // 2 - ws:t.shape[2] // 2 + ws, t.shape[3] // 2 - ws:t.shape[3] // 2 + ws] #* 1.0
                t_cc[:, self.map_index,...] = 1

                if self.method == 'maximize':
                    self.losses[module] = -(s_cc * t_cc.contiguous()).sum()
                else:
                    self.losses[module] = torch.abs(200 -(s_cc * t_cc.contiguous())).sum()


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
