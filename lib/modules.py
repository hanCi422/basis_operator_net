import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FNN(nn.Module):
    def __init__(self, hidden_layer=[64, 64], dim_in=-1, dim_out=-1, activation=None):
        super().__init__()
        self.sigma = activation
        dim = [dim_in] + hidden_layer + [dim_out]
        self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.sigma(self.layers[i](x))
        # linear activation at the last layer
        return self.layers[-1](x)

class NeuralBasis(nn.Module):
    def __init__(self, dim_in=1, hidden=[4,4,4], n_base=4, activation=None):
        super().__init__()
        self.sigma = activation
        dim = [dim_in] + hidden + [n_base]
        self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])

    def forward(self, t):
        for i in range(len(self.layers)-1):
            t = self.sigma(self.layers[i](t))
        # linear activation at the last layer
        return self.layers[-1](t)


class Basic_Model(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

    def load_params_from_file(self, filename, optimizer=None, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        print('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        if optimizer:
            optimizer_state_disk = checkpoint['optimizer_state']
            optimizer.load_state_dict(optimizer_state_disk)
            print('loaded optimizer')
        else:
            print('optimizer is not loaded')

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                print('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                print('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        print('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))
    
    def freeze_basis(self, lr=1e-4, weight_decay=0):
        total = 0
        freezed = 0
        for name, param in self.named_parameters():
            if name[:5] == 'BL_in':
                param.requires_grad = False
                freezed += 1
                print('Freeze '+name)
            total += 1
        print('==> Freezed (loaded %d/%d)' % (freezed, total))
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)
        return optimizer