import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../../')
from lib.modules import *
import numpy as np
from sklearn.neighbors import KernelDensity

class BasisONet(Basic_Model):
    def __init__(self, n_base_in=9, base_in_hidden=[64, 64, 64], middle_hidden=[64, 64, 64], \
        n_base_out=9, base_out_hidden=[64, 64, 64], grid_in=None, grid_out=None, device=None, activation=None):
        super().__init__()
        self.n_base_in = n_base_in
        self.n_base_out = n_base_out
        self.device = device
        self.t_in = torch.tensor(grid_in).to(device).float().reshape(-1, 2)
        self.t_out = torch.tensor(grid_out).to(device).float().reshape(-1, 2)
        self.BL_in = NeuralBasis(2, hidden=base_in_hidden, n_base=n_base_in, activation=activation)
        self.Middle = FNN(hidden_layer=middle_hidden, dim_in=n_base_in, dim_out=n_base_out, activation=activation)
        self.BL_out = NeuralBasis(2, hidden=base_out_hidden, n_base=n_base_out, activation=activation)
        self.kde_in = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(grid_in)
        self.density_in = np.exp(self.kde_in.score_samples(grid_in))
        self.density_in = torch.tensor(self.density_in).to(device).float().reshape(-1, 1)
        self.kde_out = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(grid_out)
        self.density_out = np.exp(self.kde_out.score_samples(grid_out))
        self.density_out = torch.tensor(self.density_out).to(device).float().reshape(-1, 1)

    def forward(self, x, y):
        B_in, J1_in = x.size()
        x = x.reshape(B_in, -1)
        B_out, J1_out = y.size()
        y = y.reshape(B_out, -1)
        T_in, T_out = self.t_in, self.t_out
        density_in = self.density_in
        density_in = density_in.squeeze(1)
        density_out = self.density_out
        density_out = density_out.squeeze(1)
        self.bases_in = self.BL_in(T_in) # (J1_in*J2_out, n_base_in)
        self.bases_out = self.BL_out(T_out) # (J1_out*J2_out, n_base_out)
        score_in = torch.einsum('bs,sn->bn', x/density_in, self.bases_in) / self.bases_in.shape[0] # (B, n_base_in)
        score_out = self.Middle(score_in) # (B, n_basis_out)
        out = torch.einsum('bn,sn->bs', score_out, self.bases_out) # (B, J1_out*J2_out*J3_out)
        autoencoder_in = torch.einsum('bn,sn->bs', score_in, self.bases_in)
        # density_out = self.Density_net_out(self.density_out)
        # y_tmp = y / self.density_out.squeeze(1)
        score_out_temp = torch.einsum('bs,sn->bn', y/density_out, self.bases_out) / self.bases_out.shape[0]
        autoencoder_out = torch.einsum('bn,sn->bs', score_out_temp, self.bases_out)
        return out, autoencoder_in, autoencoder_out

    def check_orthogonality_in(self, path=None):
        T = self.t_in
        # evaluate the current basis nodes at time grid
        self.bases_in = self.BL_in(T) # (J, n_base)
        density = self.density_in
        density = density.squeeze(1)
        orth_matrix = torch.ones((self.bases_in.shape[1], self.bases_in.shape[1])).to(self.device)
        for i in range(orth_matrix.shape[0]):
            for j in range(orth_matrix.shape[1]):
                orth_matrix[i, j] = torch.einsum('s,s->', self.bases_in[:, i]/density, self.bases_in[:, j]) / self.bases_in.shape[0]
        orth_matrix = orth_matrix.detach().cpu().numpy()
        if path:
            np.savetxt(path, orth_matrix)
        else:
            return orth_matrix
    def check_orthogonality_out(self, path=None):
        T = self.t_out
        # evaluate the current basis nodes at time grid
        self.bases_out = self.BL_out(T) # (J, n_base)
        orth_matrix = torch.ones((self.bases_out.shape[1], self.bases_out.shape[1])).to(self.device)
        density = self.density_out
        density = density.squeeze(1)
        for i in range(orth_matrix.shape[0]):
            for j in range(orth_matrix.shape[1]):
                orth_matrix[i, j] = torch.einsum('s,s->', self.bases_out[:, i]/density, self.bases_out[:, j]) / self.bases_out.shape[0]
        orth_matrix = orth_matrix.detach().cpu().numpy()
        if path:
            np.savetxt(path, orth_matrix)
        else:
            return orth_matrix