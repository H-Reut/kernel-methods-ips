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


# plotting μ as 2d heatmap:


def plot(μ, t_index):
    plt.imshow(μ[t_index, :, :], extent=[v_start, v_end, x_start, x_end], aspect='auto', origin='lower')
    plt.xlabel('velocity $v$')
    plt.ylabel('position $x$')
    plt.title(f'$\\mu(t,x,v)$ at time $t={t[t_index]:.3f}$, timestep: {t_index}, mass: {np.sum(μ[0])*intFac}')
    #plt.colorbar(label='$\\mu$ value')
    #plt.gca().set_aspect('equal')
    plt.show()


########## Solving positions (x) and velocities (v) ##########
start_time = time.time()

Δt2Δx = Δt / (2 * Δx)
Δt2Δv = Δt / (2 * Δv)
#H1 = H_β(x[:, np.newaxis] - x[np.newaxis, :])
#wv = v[:, np.newaxis] - v[np.newaxis, :]
for n in range(N_t-1):
    if n == 100:#n%100 == 0:
        plot(μ, n)
    print(f"\tsolving time step:\t{str(n+1).rjust(len(str(N_t-1)))} / {N_t-1}\t({(n+1)/(N_t-1):.0%})", end="\r")
    if n==50:
        print(
            f"\nTime elapsed after 50 steps: {time.time() - start_time:.2f} seconds")
    #h_n = np.zeros((N_x, N_v))
    #def h(x, v):
    #    print(x.shape)
    #    diffx = x[np.newaxis, :] - x[:]
    #    print(diffx.shape)
    #    #temp = H_β(
    
    # solving μ
    a = (μ[n, 0:-2, 1:-1] + μ[n, 2:, 1:-1] + μ[n, 1:-1, 0:-2] + μ[n, 1:-1, 2:]) / 4
    b = Δt2Δx * (v[1:-1][np.newaxis, :] * (μ[n, 2:, 1:-1] - μ[n, 0:-2, 1:-1]))
    μ[n+1, 1:-1, 1:-1] = a - b
    h_ij = np.zeros((N_x-2, N_v))
#    h_M1 = np.zeros((N_x-2, N_v))
    for i in range(1, N_x-1):
        #print(i)
        h_yx  = y_x[N_x-1-i : N_x+N_x-1-i]
        h_H_β = H_β(h_yx)[:,np.newaxis] * μ[n]
        for j in range(0, N_v):
            h_ij[i-1, j] = (h_H_β * w_v[N_v-1-j : N_v+N_v-1-j]).sum()
            # toDo: parallelize even more such that the i- and j- loops only contain
            # a single statement, e.g.:  h_ij[i, j] = A[i:i+N, j:j+N].sum()
            # perhaps use np.cumsum() or other form of rolling sum
            # runtime of current implementation with N_x=N_v=N_t=1000 is ~30 days
    h2 = h_ij[:] * μ[n, 1:-1, :]
    μ[n+1, 1:-1, 1:-1] -= Δt2Δv * intFac * (h2[:, 2:] - h2[:, :-2])
    μ[n+1,0] = μ[n+1,1].copy()
    μ[n+1,N_x-1] = μ[n+1,N_x-2].copy()
    μ[n+1,:,0] = μ[n+1,:,1].copy()
    μ[n+1,:,N_v-1] = μ[n+1,:,N_v-2].copy()
plot(μ, -1)
