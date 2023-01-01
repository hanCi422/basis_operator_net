import os
import numpy as np
import torch

class Config(object):
    ntrain = 1000
    nvalid = 1000
    ntest = 1000
    nbasis_in = 9
    nbasis_out = 9
    batch_size = 100
    sub = 1
    learning_rate = 1e-3
    epochs = 20000
    base_in_hidden = [512, 512, 512, 512, 512]
    base_out_hidden = [512, 512, 512, 512, 512]
    middle_hidden = [512, 512, 512]
    model_name = 'BasisONet'
    activation = None
    lambda_in = 1.0
    lambda_out = 1.0
    device = 'cuda'


class LpLoss(object):
    def __init__(self, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

def _inner_product(f1, f2, h):
    """    
    f1 - (B, J) : B functions, observed at J time points,
    f2 - (B, J) : same as f1
    h  - (J-1,1): weights used in the trapezoidal rule
    pay attention to dimension
    <f1, f2> = sum (h/2) (f1(t{j}) + f2(t{j+1}))
    """
    prod = f1 * f2 # (B, J = len(h) + 1)
    return torch.matmul((prod[:, :-1] + prod[:, 1:]), h.unsqueeze(dim=-1))/2

def _parralleled_inner_product(f1, f2, h):
    prod = f1 * f2
    return torch.matmul((prod[:, :, :-1] + prod[:, :, 1:]), h)/2

def trapezoidal_2d_parralleled(f, h):
    assert isinstance(h, list) and len(h) == 2
    _, _, l1, l2 = f.size()
    # l1, l2 = f.shape[2], f.shape[3]
    c = torch.ones((l1, l2),device=f.device)
    c[[0,-1], :] = 1/2
    c[:, [0,-1]] = 1/2
    c[[0,0,-1,-1], [0,-1,0,-1]] = 1/4
    return h[0] * h[1] * torch.sum(torch.mul(c, f), dim=(-2, -1))

def _parralleled_inner_product_2d(f1, f2, h):
    prod = f1 * f2
    return trapezoidal_2d_parralleled(prod, h)

def trapezoidal_2d(f, h):
    assert isinstance(h, list) and len(h) == 2
    _, l1, l2 = f.size()
    # l1, l2 = f.shape[1], f.shape[2]
    c = torch.ones((l1, l2),device=f.device)
    c[[0,-1], :] = 1/2
    c[:, [0,-1]] = 1/2
    c[[0,0,-1,-1], [0,-1,0,-1]] = 1/4
    return h[0] * h[1] * torch.sum(torch.mul(c, f), dim=(-2, -1))

def _inner_product_2d(f1, f2, h):
    prod = f1 * f2
    return trapezoidal_2d(prod, h)

class Logger():
    def __init__(self, subpath):
        if not os.path.exists('logs'):
            os.mkdir('logs')
        self.logger = open(os.path.join('logs', subpath + '.txt'), 'w')
    def log_string(self, out_str):
        self.logger.write(out_str+'\n')
        self.logger.flush()
        print(out_str)