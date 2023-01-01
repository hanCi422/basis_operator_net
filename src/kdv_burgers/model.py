import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../../')
from lib.modules import *
from lib.utils import _inner_product, _parralleled_inner_product, _parralleled_inner_product_2d, _inner_product_2d
import numpy as np

class BasisONet(Basic_Model):
    def __init__(self, n_base_in=9, base_in_hidden=[64, 64, 64], middle_hidden=[64, 64, 64], \
        n_base_out=9, base_out_hidden=[64, 64, 64], grid_in=None, grid_out=None, device=None, activation=None):
        super().__init__()
        self.n_base_in = n_base_in
        self.n_base_out = n_base_out
        self.device = device
        self.t_in = torch.tensor(grid_in).to(device).float()
        self.t_out = torch.tensor(grid_out).to(device).float().reshape(-1, 2)
        self.h_in = torch.tensor(grid_in[1:] - grid_in[:-1]).to(device).float()
        self.h_out = [grid_out[0,1,0] - grid_out[0,0,0], grid_out[1,0,1] - grid_out[0,0,1]]
        self.BL_in = NeuralBasis(1, hidden=base_in_hidden, n_base=n_base_in, activation=activation)
        self.Middle = FNN(hidden_layer=middle_hidden, dim_in=n_base_in, dim_out=n_base_out, activation=activation)
        self.BL_out = NeuralBasis(2, hidden=base_out_hidden, n_base=n_base_out, activation=activation)

    def forward(self, x, y):
        B_in, J_in = x.size()
        B_out, J1_out, J2_out = y.size()
        T_in, T_out = self.t_in.unsqueeze(dim=-1), self.t_out
        self.bases_in = self.BL_in(T_in) # (J_in, n_base_in)
        self.bases_out = self.BL_out(T_out) # (J1_out*J2_out, n_base_out)
        self.bases_in = self.bases_in.transpose(-1, -2) # (n_base_in, J_in)
        self.bases_out = self.bases_out.transpose(-1, -2) # (n_base_out, J1_out*J2_out)
        score_in = _parralleled_inner_product(x.unsqueeze(1).repeat((1, self.n_base_in, 1)), \
            self.bases_in.unsqueeze(0).repeat((B_in, 1, 1)), self.h_in) # (B_in, n_base_in)
        score_out = self.Middle(score_in)
        out = torch.mm(score_out, self.bases_out)
        autoencoder_in = torch.mm(score_in, self.bases_in)
        score_out_temp = _parralleled_inner_product_2d(y.unsqueeze(1).repeat((1, self.n_base_out, 1, 1)), \
            self.bases_out.unsqueeze(0).repeat((B_out, 1, 1)).reshape(B_out, self.n_base_out, J1_out, J2_out), self.h_out) # (B_out, n_base_out)
        autoencoder_out = torch.mm(score_out_temp, self.bases_out)
        return out, autoencoder_in, autoencoder_out
    
    def aec_in_forward(self, x):
        B, J = x.size()
        assert J == self.h.size()[0] + 1
        T = self.t_in.unsqueeze(dim=-1)
        # evaluate the current basis nodes at time grid
        self.bases_in = self.BL_in(T) # (J, n_base)
        self.bases_in = self.bases_in.transpose(-1, -2) # (n_base, J)
        score_in = _parralleled_inner_product(x.unsqueeze(1).repeat((1, self.n_base_in, 1)), \
            self.bases_in.unsqueeze(0).repeat((B, 1, 1)), self.h_in) # (B, n_base)
        autoencoder = torch.mm(score_in, self.bases_in) # (B, n_grid)
        return autoencoder
    
    def aec_out_forward(self, y):
        B, J1, J2 = y.size()
        # evaluate the current basis nodes at time grid
        self.bases_out = self.BL_out(self.t_out) # (J, n_base)
        self.bases_out = self.bases_out.transpose(-1, -2) # (n_base, J1*J2)
        score_out =  _parralleled_inner_product_2d(y.unsqueeze(1).repeat((1, self.n_base_out, 1, 1)), \
            self.bases_out.unsqueeze(0).repeat((B, 1, 1)).reshape(B, self.n_base_out, J1, J2), self.h_out)
        autoencoder = torch.mm(score_out, self.bases_out) # (B, n_grid)
        return autoencoder

    def check_orthogonality_in(self, path=None):
        T = self.t_in.unsqueeze(dim=-1)
        # evaluate the current basis nodes at time grid
        self.bases_in = self.BL_in(T) # (J, n_base)
        self.bases_in = self.bases_in.transpose(-1, -2) # (n_base, J)
        orth_matrix = torch.ones((self.bases_in.shape[0], self.bases_in.shape[0])).to(self.device)
        for i in range(orth_matrix.shape[0]):
            for j in range(orth_matrix.shape[1]):
                orth_matrix[i, j] = _inner_product(self.bases_in[i, :].unsqueeze(0), \
                    self.bases_in[j, :].unsqueeze(0), self.h_in).squeeze()
        orth_matrix = orth_matrix.detach().cpu().numpy()
        if path:
            np.savetxt(path, orth_matrix)
        else:
            return orth_matrix

    def check_orthogonality_out(self, J1, J2, path=None):
        # evaluate the current basis nodes at time grid
        self.bases_out = self.BL_out(self.t_out) # (J, n_base)
        self.bases_out = self.bases_out.transpose(-1, -2) # (n_base, J1*J2)
        orth_matrix = torch.ones((self.bases_out.shape[0], self.bases_out.shape[0])).to(self.device)
        for i in range(orth_matrix.shape[0]):
            for j in range(orth_matrix.shape[1]):
                orth_matrix[i, j] = _inner_product_2d(self.bases_out[i, :].reshape(J1, J2).unsqueeze(0), \
                    self.bases_out[j, :].reshape(J1, J2).unsqueeze(0), self.h_out).squeeze()
        orth_matrix = orth_matrix.detach().cpu().numpy()
        if path:
            np.savetxt(path, orth_matrix)
        else:
            return orth_matrix

