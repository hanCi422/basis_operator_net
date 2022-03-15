import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
import os

def Diff_mat_1D(Nx):
    # First derivative
    D_1d = sp.diags([-1, 1], [-1, 1], shape = (Nx,Nx)) # A division by (2*dx) is required later.
    D_1d = sp.lil_matrix(D_1d)
    D_1d[0,[0,1,2]] = [-3, 4, -1]               # this is 2nd order forward difference (2*dx division is required)
    D_1d[Nx-1,[Nx-3, Nx-2, Nx-1]] = [1, -4, 3]  # this is 2nd order backward difference (2*dx division is required)
    # Second derivative
    D2_1d =  sp.diags([1, -2, 1], [-1,0,1], shape = (Nx, Nx)) # division by dx^2 required
    D2_1d = sp.lil_matrix(D2_1d)                  
    D2_1d[0,[0,1,2,3]] = [2, -5, 4, -1]                    # this is 2nd order forward difference. division by dx^2 required. 
    D2_1d[Nx-1,[Nx-4, Nx-3, Nx-2, Nx-1]] = [-1, 4, -5, 2]  # this is 2nd order backward difference. division by dx^2 required.
    return D_1d, D2_1d

def Diff_mat_2D(Nx,Ny):
    # 1D differentiation matrices
    Dx_1d, D2x_1d = Diff_mat_1D(Nx)
    Dy_1d, D2y_1d = Diff_mat_1D(Ny)
    # Sparse identity matrices
    Ix = sp.eye(Nx)
    Iy = sp.eye(Ny)
    # 2D matrix operators from 1D operators using kronecker product
    # First partial derivatives
    Dx_2d = sp.kron(Iy,Dx_1d)
    Dy_2d = sp.kron(Dy_1d,Ix)
    # Second partial derivatives
    D2x_2d = sp.kron(Iy,D2x_1d)
    D2y_2d = sp.kron(D2y_1d,Ix)
    # Return compressed Sparse Row format of the sparse matrices
    return Dx_2d.tocsr(), Dy_2d.tocsr(), D2x_2d.tocsr(), D2y_2d.tocsr()

def sin_basis_2d(grid_x, grid_y, nbasis, num):
    def _sin_basis_2d(x1, x2, u, w):
        return np.sqrt(2) * np.sin(np.pi*(x1 * u + x2 * w))
    assert isinstance(nbasis, list) and len(nbasis) == 2
    w, h = grid_x.shape[0], grid_x.shape[1]
    results = np.zeros_like(grid_x)[None, :]
    results = np.repeat(results, num, axis=0)
    for i in range(nbasis[0]):
        for j in range(nbasis[1]):
            coef = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(num, 1, 1))
            coef = np.repeat(coef, w, axis=1)
            coef = np.repeat(coef, h, axis=2)
            if i == 0 and j == 0:
                results += coef
            else:
                results += coef * _sin_basis_2d(grid_x, grid_y, i, j)
    return results

# Dirichlet boundary conditions
uL = 0
uR = 0
uT = 0
uB = 0

# Define independent variables
Nx = 101                    # No. of grid points along x direction
Ny = 101                        # No. of grid points along y direction
x = np.linspace(0,1,Nx)        # x variables in 1D
y = np.linspace(0,1,Ny)        # y variable in 1D

dx = x[1] - x[0]                # grid spacing along x direction
dy = y[1] - y[0]                # grid spacing along y direction

X,Y = np.meshgrid(x,y)          # 2D meshgrid

# 1D indexing
Xu = X.ravel()                  # Unravel 2D meshgrid to 1D array
Yu = Y.ravel()

# Loading finite difference matrix operators
Dx_2d, Dy_2d, D2x_2d, D2y_2d = Diff_mat_2D(Nx,Ny)   # Calling 2D matrix operators from funciton

# Boundary indices
start_time = time.time()
ind_unravel_L = np.squeeze(np.where(Xu==x[0]))          # Left boundary
ind_unravel_R = np.squeeze(np.where(Xu==x[Nx-1]))       # Right boundary
ind_unravel_B = np.squeeze(np.where(Yu==y[0]))          # Bottom boundary
ind_unravel_T = np.squeeze(np.where(Yu==y[Ny-1]))       # Top boundary
ind_boundary_unravel = np.squeeze(np.where((Xu==x[0]) | (Xu==x[Nx-1]) | (Yu==y[0]) | (Yu==y[Ny-1])))  # All boundary
ind_boundary = np.where((X==x[0]) | (X==x[Nx-1]) | (Y==y[0]) | (Y==y[Ny-1]))    # All boundary

# Construction of the system matrix
start_time = time.time()
I_sp = sp.eye(Nx*Ny).tocsr()
L_sys = D2x_2d/dx**2 + D2y_2d/dy**2     # system matrix without boundary conditions
L_sys[ind_boundary_unravel,:] = I_sp[ind_boundary_unravel,:]

in_f = []
out_f = []
nsample = 3000
nbasis = [5, 5]
for i in tqdm(range(nsample)):
    # Source function (right hand side vector)
    g = sin_basis_2d(X, Y, nbasis, 1).reshape(-1, Nx*Ny)
    g = np.squeeze(g)
    # Construction of right hand vector (function of x and y)
    b = g.copy()
    b[ind_unravel_L] = uL
    b[ind_unravel_R] = uR
    b[ind_unravel_T] = uT
    b[ind_unravel_B] = uB
    # solve
    start_time = time.time()
    u = spsolve(L_sys,b).reshape(Ny,Nx)[None, :, :]
    in_f.append(g.reshape(Nx, Ny)[None, :, :])
    out_f.append(u)

in_f = np.vstack(in_f)
out_f = np.vstack(out_f)
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*[x, y])]).T

save_path = 'datasets/sin_5_5'
if not os.path.exists(save_path):
    os.makedirs(save_path)
np.save(os.path.join(save_path, 'in_f.npy'), in_f)
np.save(os.path.join(save_path, 'out_f.npy'), out_f)
np.save(os.path.join(save_path, 'grid.npy'), grid.reshape(Nx, Ny, 2))
