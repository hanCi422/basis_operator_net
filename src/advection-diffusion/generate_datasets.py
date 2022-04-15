# modify from https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets

import jax
import jax.numpy as np
from jax import random, vmap
from jax.config import config
from jax.ops import index_update, index
from jax import lax
from tqdm import tqdm
import os

# Use double precision to generate data (due to GP sampling)
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale * np.exp(-0.5 * r2)

f = None
g = None

# Advection-diffusion solver 
def solve_ADVD(key, gp_sample, Nx, Nt, m, P):
    """Solve
    u_t + u_x - D * u_xx = 0
    u(x, 0) = V(x)
    """
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    D = 0.1
    N = gp_sample.shape[0]
    X = np.linspace(xmin, xmax, N)[:,None]
    V = lambda x: np.interp(x, X.flatten(), gp_sample)

    # Create grid
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    lam = dt / h
    mu = dt / h ** 2

    v_fn = lambda x: V(np.sin(np.pi * x) ** 2)
    v =  v_fn(x)

    u = np.zeros([Nx, Nt])
    u = index_update(u, index[:, 0], v)

    I = np.eye(Nx - 1)
    I1 = np.roll(I, 1, axis=0)
    I2 = np.roll(I, -1, axis=0)
    A = (1 + D * mu) * I - (lam / 4 + D * mu / 2) * I1 + (lam / 4 - D * mu / 2) * I2
    B = 2 * I - A
    C = np.linalg.solve(A, B)
    
    def body_fn_t(i, u):
        u = index_update(u, index[1:, i + 1], C @ u[1:, i])
        return u
    UU = lax.fori_loop(0, Nt-1, body_fn_t, u)
    UU = index_update(UU, index[0, :], UU[-1, :])
    # Input sensor locations and measurements
    xx = np.linspace(xmin, xmax, m)
    u = v_fn(xx)
    # Output sensor locations and measurements
    idx = random.randint(key, (P,2), 0, max(Nx,Nt))
    y = np.concatenate([x[idx[:,0]][:,None], t[idx[:,1]][:,None]], axis = 1)
    s = UU[idx[:,0], idx[:,1]]

    return (x, t, UU), (u, y, s)

def generate_one_data(key, Nx, Nt, P):
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = np.linspace(xmin, xmax, N)[:,None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter*np.eye(N))
    gp_sample = np.dot(L, random.normal(key, (N,)))

    (x, t, UU), (u, y, s) = solve_ADVD(key, gp_sample, Nx, Nt, m, P)

    XX, TT = np.meshgrid(x, t)

    u_test = np.tile(u, (Nx*Nt,1))
    y_test = np.hstack([XX.flatten()[:,None], TT.flatten()[:,None]])
    s_test = UU.T.flatten()

    return u_test, y_test, s_test

def generate_data(key, N, Nx, Nt, P):

    config.update("jax_enable_x64", True)
    keys = random.split(key, N)

    u_test, y_test, s_test = vmap(generate_one_data, (0, None, None, None))(keys, Nx, Nt, P)

    u_test = np.float32(u_test.reshape(N * Nx * Nt,-1))
    y_test = np.float32(y_test.reshape(N * Nx * Nt,-1))
    s_test = np.float32(s_test.reshape(N * Nx * Nt,-1))

    config.update("jax_enable_x64", False)
    return u_test, y_test, s_test

xmin, xmax = 0, 1
tmin, tmax = 0, 1
length_scale = 0.5
Nx = 101
Nt = 101
m = Nx
P_test = 101
key = random.PRNGKey(0)

nsample = 3000
in_f = []
out_f = []
keys = random.split(key, nsample)
for i in tqdm(range(nsample)):
    u_test, y_test, s_test = generate_data(keys[i], 1, Nx, Nt, P_test)
    in_f.append(u_test[0, :][None, :])
    out_f.append(s_test.reshape(1, m, P_test))
in_f = np.vstack(in_f)
out_f = np.vstack(out_f)

save_path = 'datasets/grf_0.5'
if not os.path.exists(save_path):
    os.makedirs(save_path)
np.save(os.path.join(save_path, 'in_f.npy'), in_f)
np.save(os.path.join(save_path, 'out_f.npy'), out_f)
np.save(os.path.join(save_path, 'grid_in.npy'), np.linspace(0, 1, m))
np.save(os.path.join(save_path, 'grid_out.npy'), y_test.reshape(m, P_test, 2))