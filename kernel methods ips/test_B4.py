from math import factorial
from tkinter import N
from _pytest.monkeypatch import V
import numpy as np
import matplotlib.pyplot as plt
import shared_functions
import time

# repeatable randomness
seed = np.random.randint(2147483647)
print(f'test_B1.py\t\tseed:\t{seed}')
rng = np.random.default_rng(seed=seed)

np.set_printoptions(linewidth=250)

########## Parameters ##########
# time dimension
t_start = 0.0
t_end   = 1.0
N_t     = 1000
Δt      = (t_end - t_start) / N_t
t       = np.linspace(t_start, t_end, N_t)

# space dimension
x_start = 0.0
x_end   = 3.0
N_x     = 100#0
Δx      = (x_end - x_start) / N_x
x       = np.linspace(x_start, x_end, N_x)
y_x     = np.linspace(x_start-x_end, x_end-x_start, 2*N_x-1)
# w-v for all w and given v (or rather: given j, v=v[j])
# w-0 = w is in { v[0], ..., v[N_v-1] } = { w_v[N_v-1], ..., w_v[N_v-1 + N_v -1] }
# w-1 is in { v[-1], ..., v[0] }
# w-v[j] has indices [0-j], ..., [N_v-1-j] in v

# velocity dimension
v_start = 0.0
v_end   = 3.0
N_v     = 100#0
Δv      = (v_end - v_start) / N_v
v       = np.linspace(v_start, v_end, N_v)
w_v     = np.linspace(v_start-v_end, v_end-v_start, 2*N_v-1)  # v[i] = w_v[i + N_x - 1]

# distribution
μ       = np.zeros((N_t, N_x, N_v))
A       = (x_end-x_start) * (v_end-v_start)  # Area
intFac  = A/N_x/N_v
# to calculate a double integral ∫∫(⋅)dxdv we can sum over the grid and multiply by intFac


# initial values: μ(t[0]) ~ 𝒰([1,2]×[1,2])
'''μ[0, 333:666, 333:666] = np.ones((333, 333))
μ[0, 333:666, 666] = 1/3*np.ones(333)
μ[0, 666, 333:666] = 1/3*np.ones(333)
μ[0, 666, 666] = 1/9'''

μ[0, 33:66, 33:66] = np.ones((33, 33))
μ[0, 33:66, 66] = 1/3*np.ones(33)
μ[0, 66, 33:66] = 1/3*np.ones(33)
μ[0, 66, 66] = 1/9

'''μ[0, 3:6, 3:6] = np.ones((3, 3))
μ[0, 3:6, 6] = 1/3*np.ones(3)
μ[0, 6, 3:6] = 1/3*np.ones(3)
μ[0, 6, 6] = 1/9'''
print(f'mass in μ:\t∫∫ μ(x,v) dx dv = {np.sum(μ[0])*intFac}')


# model parameters
β = 2.0             # parameter of H_β
γ = 1.0/np.sqrt(2)  # parameter of k_γ

# interpolation parameter
s = 4       # number of time samples

# Interaction function H_β(x-xʹ) for Cucker-Smale systems
def H_β(diff, β=2.0):
    # As part of H_β we have to calculate the 2-norm  ||x − xʹ||
    # Instead of H_β(x, xʹ), we implement H_β(diff) which must be called with diff=x-xʹ
    # The model is 1d, and np.linalg.norm() doesn't work on scalars, so instead we use np.abs()
    return 1 / (1 + np.abs(diff)**2)**β



def max_(μ_t):
    flat_index = np.argmax(μ_t)
    ij_max = np.unravel_index(flat_index, μ_t.shape)
    return ij_max / np.array([N_x, N_v]) * np.array([x_end-x_start, v_end-v_start])


def mean(μ_t):
    # create grid of (x, v) pairs: xv[i, j] = [x[i], v[j]]
    X, V = np.meshgrid(x, v, indexing='ij')  # X, V shapes: (N_x, N_v)
    xv = np.stack((X, V), axis=-1)           # xv shape: (N_x, N_v, 2)
    #print(xv)
    print(np.sum(xv * μ_t[..., np.newaxis], axis=(0, 1)) * intFac)
    return np.sum(xv * μ_t[..., np.newaxis], axis=(0, 1)) * intFac


def variance(μ_t):
    # create grid of (x, v) pairs: xv[i, j] = [x[i], v[j]]
    X, V = np.meshgrid(x, v, indexing='ij')  # X, V shapes: (N_x, N_v)
    xv = np.stack((X, V), axis=-1)           # xv shape: (N_x, N_v, 2)
    mean_μ_t = mean(μ_t)
    integral = np.linalg.norm(xv - mean_μ_t[np.newaxis, np.newaxis, :], axis=(-1))**2 * μ_t
    print(integral.sum() * intFac)
    return integral.sum() * intFac


def skewness(μ_t):
    # create grid of (x, v) pairs: xv[i, j] = [x[i], v[j]]
    X, V = np.meshgrid(x, v, indexing='ij')  # X, V shapes: (N_x, N_v)
    xv = np.stack((X, V), axis=-1)           # xv shape: (N_x, N_v, 2)
    mean_μ_t = mean(μ_t)
    var_μ_t = variance(μ_t)


# plotting μ as 2d heatmap:
def plot(μ, t_index):
    plt.imshow(μ[t_index, :, :], extent=[v_start, v_end, x_start, x_end], aspect='auto', origin='lower')

    '''max_μ = (max_(μ[t_index]))[::-1]
    mean_μ = (mean(μ[t_index]))[::-1]
    var_μ = variance(μ[t_index])
    plt.plot(*max_μ, 'kx', fillstyle='none', label='max')
    plt.plot(*mean_μ, 'r+', fillstyle='none', label='mean')
    var_circle = plt.Circle(mean_μ, var_μ, color='r', fill=False, label='variance')
    plt.gca().add_patch(var_circle)'''

    plt.xlabel('velocity $v$')
    plt.ylabel('position $x$')
    plt.title(f'$\\mu(t,x,v)$ at time $t={t[t_index]:.3f}$, timestep: {t_index}, mass: {np.sum(μ[0])*intFac}')
    #plt.colorbar(label='$\\mu$ value')
    plt.gca().set_aspect('equal')
    #plt.legend()    
    plt.show()


########## Solving positions (x) and velocities (v) ##########
# toDo: parallelize even more such that the i- and j- loops only contain
# a single statement, e.g.:  h_ij[i, j] = A[i:i+N, j:j+N].sum()
# runtime of current implementation with N_x=N_v=N_t=1000 is ~30 days
def hLoop1():
    h_ij = np.zeros((N_x, N_v))#(N_x-2, N_v))
    for i in range(N_x):#(1, N_x-1):
        h_yx  = y_x[N_x-1-i : N_x+N_x-1-i]
        h_H_β = H_β(h_yx)[:,np.newaxis] * μ[n]
        for j in range(0, N_v):
            #h_ij[i-1, j] = (h_H_β * w_v[N_v-1-j : N_v+N_v-1-j]).sum()
            h_ij[i, j] = (h_H_β * w_v[N_v-1-j : N_v+N_v-1-j]).sum()
    h = h_ij[:] * μ[n, :, :]#μ[n, 1:-1, :]
    return h

def hLoop1b():
    h_ij = np.zeros((N_x, N_v))
    y_x_i = np.lib.stride_tricks.sliding_window_view(y_x, N_x)[::-1]
    h_yx_i= H_β(np.lib.stride_tricks.sliding_window_view(y_x, N_x))[::-1]
    w_v_j = np.lib.stride_tricks.sliding_window_view(w_v, N_v)[::-1]
    for i in range(N_x):
        h_H_β = H_β(y_x_i[i])[:,np.newaxis] * μ[n]
        for j in range(0, N_v):
            h_ij[i, j] = (h_H_β * w_v_j[j]).sum()
    h = h_ij[:] * μ[n, :, :]
    return h

def hLoop1b2():
    for i in range(N_x):
        h_H_β = H_β_y_x_i[i,:,np.newaxis] * μ[n]
        for j in range(0, N_v):
            h_ij[i, j] = (h_H_β * w_v_j[j]).sum(axis=(-2, -1))
    h = h_ij[:, :] * μ[n, :, :]
    return h

def hLoop1b3(): # fastest so far
    h_H_β = H_β_y_x_i[:,:,np.newaxis] * μ[np.newaxis, n, :, :]
    for i in range(N_x):
        for j in range(0, N_v):
            h_ij[i, j] = (h_H_β[i] * w_v_j[j]).sum(axis=(-2, -1))
    h = h_ij[:, :] * μ[n, :, :]
    return h

def hLoop1c():
    h_ij = np.zeros((N_x, N_v))
    h_yx_i = np.lib.stride_tricks.sliding_window_view(y_x, N_x)[::-1]
    w_v_j = np.lib.stride_tricks.sliding_window_view(w_v, N_v)[::-1]
    for i in range(N_x):
        h_H_β = H_β(h_yx_i[i])[:,np.newaxis] * μ[n]
        h_ij[i, :] = (h_H_β[np.newaxis, :, :] * w_v_j[:, np.newaxis, :]).sum(axis=(-1, -2))
    h = h_ij[:] * μ[n, :, :]
    return h

def hLoop2():
    h_ij = np.zeros((N_x, N_v))
    H_β_yx = H_β(y_x)[:,np.newaxis] 
    for i in range(N_x):
        h_H_β = H_β_yx[N_x-1-i : N_x+N_x-1-i] * w_v[np.newaxis, :]
        for j in range(0, N_v):
            h_ij[i, j] = (h_H_β[:, N_v-1-j : N_v+N_v-1-j] * μ[n]).sum()
    h = (h_ij[:] * μ[n, :, :])
    return h

def hNpArrArr():
    H_β_ij = H_β(y_x)[:,np.newaxis] * w_v
    Integrand = np.lib.stride_tricks.sliding_window_view(H_β_ij, (N_x, N_v))
    Integral = (Integrand * μ[n, np.newaxis, np.newaxis, :, :]).sum(axis=(-1, -2))
    h = Integral[::-1, ::-1] * μ[n]
    return h

def hNpLoopArr():
    H_β_ij = H_β(y_x)[:,np.newaxis] * w_v
    Integrand = np.lib.stride_tricks.sliding_window_view(H_β_ij, (N_x, N_v))
    Integral = np.zeros((N_x, N_v))
    for i in range(N_x-1):
        Integral[i,:] = (Integrand[i,:] * μ[n, np.newaxis, :, :]).sum(axis=(-1, -2))
    h = Integral[::-1, ::-1] * μ[n]
    return h

def hNpLoopLoop():
    H_β_ij = H_β(y_x)[:,np.newaxis] * w_v
    Integrand = np.lib.stride_tricks.sliding_window_view(H_β_ij, (N_x, N_v))
    Integral = np.zeros((N_x, N_v))
    for i in range(N_x-1):
        for j in range(N_v-1):
            Integral[i,j] = (Integrand[i,j] * μ[n, :, :]).sum(axis=(-1, -2))
    h = Integral[::-1, ::-1] * μ[n]
    return h

def hNpLoopLoop2():
    H_β_ij = H_β(y_x)[:,np.newaxis] * w_v
    Integrand = np.lib.stride_tricks.sliding_window_view(H_β_ij, (N_x, N_v))#
    Integral = np.zeros((N_x, N_v))
    for i in range(N_x-1):
        for j in range(N_v-1):
            Integral[-i-1,-j-1] = (H_β_ij[N_x-i-1 : 2*N_x-i-1 , N_v-j-1 : 2*N_v-j-1] * μ[n, :, :]).sum(axis=(-1, -2))
            print('\n\n')
            print(H_β_ij[N_x-i-1 : 2*N_x-i-1 , N_v-j-1 : 2*N_v-j-1])
            print(Integrand[-i-1,-j-1])
            assert (Integrand[-i-1,-j-1] * μ[n, :, :]).sum(axis=(-1, -2)) == Integral[i,j]
    h = Integral[::-1, ::-1] * μ[n]
    return h




hfunc = hLoop1b3
print(f'Function used for calculating h:\t{hfunc.__name__}')

h_ij = np.zeros((N_x, N_v))
# pseudo-code: H_β_y_x_i[i] = [H_β(y-x[i]) for all y in position-domain]
H_β_y_x_i = H_β(np.lib.stride_tricks.sliding_window_view(y_x, N_x))[::-1]
# pseudo-code: w_v_j[j] = [(w-v[j]) for all w in velocity-domain]
w_v_j = np.lib.stride_tricks.sliding_window_view(w_v, N_v)[::-1]



start_time = time.time()
for n in range(N_t-1):
    print(f"\tsolving time step:\t{str(n+1).rjust(len(str(N_t-1)))} / {N_t-1}\t({(n+1)/(N_t-1):.0%})\tmass: {np.sum(μ[0])*intFac}", end="\r")

    #if n%100 == 0:
    if n == 0 or n == 100:
        plot(μ, n)
    if n == 50:
        print(f"\nTime elapsed after 50 steps: {time.time() - start_time:.2f} seconds")
    
    
    μn_LF = (μ[n, 0:-2, 1:-1] + μ[n, 2:, 1:-1] + μ[n, 1:-1, 0:-2] + μ[n, 1:-1, 2:]) / 4 # Lax-Friedrich
    g = v[np.newaxis, 1:-1] * (μ[n, 2:, 1:-1] - μ[n, 0:-2, 1:-1]) # g_i+1,j - g_i-1,j
    h = intFac * hfunc() # h_i,j
    μ[n+1, 1:-1, 1:-1] = μn_LF - (Δt/(2*Δx))*g - (Δt/(2*Δv))*(h[1:-1, 2:] - h[1:-1, :-2])

    # boundary conditions: 0 outwards normal derivative
    μ[n+1,0] = μ[n+1,1].copy()
    μ[n+1,N_x-1] = μ[n+1,N_x-2].copy()
    μ[n+1,:,0] = μ[n+1,:,1].copy()
    μ[n+1,:,N_v-1] = μ[n+1,:,N_v-2].copy()
plot(μ, -1)
