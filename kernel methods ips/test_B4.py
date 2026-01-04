import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import shared_functions
import time

# matplotlib printing options
plt.rcParams['figure.dpi'] = 200            # resolution of figures in dots per inch. Default is 100
plt.rcParams['figure.figsize'] = [4.8, 3.6] # size of figures in inches. Default is [6.4, 4.8]
plt.rcParams['figure.autolayout'] = True    # auto-adjust layout to avoid elements clipping outside of figure
plt.rcParams['savefig.bbox'] = "tight"      # reduce whitespace when saving figures
# numpy printing options
np.set_printoptions(linewidth=250)


########## Parameters ##########
# time dimension
t_start = 0.0
t_end   = 1.0
N_t     = 1000
Δt      = (t_end - t_start) / N_t
t       = np.linspace(t_start, t_end, N_t+1)
print(f'Time  interval:\t[{t_start}, {t_end}]\nTime steps:\tN_t = {N_t}\tΔt= {Δt}')

# space dimension
x_start = 0.0
x_end   = 3.0
N_x     = 80
Δx      = (x_end - x_start) / N_x
x       = np.linspace(x_start, x_end, N_x+1)
y_x     = np.linspace(x_start-x_end, x_end-x_start, 2*N_x+1)
print(f'Space interval:\t[{x_start}, {x_end}]\nSpac steps:\tN_x = {N_x}\tΔx= {Δx}')

# velocity dimension
v_start = 0.0
v_end   = 3.0
N_v     = N_x
Δv      = (v_end - v_start) / N_v
v       = np.linspace(v_start, v_end, N_v+1)
w_v     = np.linspace(v_start-v_end, v_end-v_start, 2*N_v+1)
print(f'Velocity intv.:\t[{v_start}, {v_end}]\nVelo steps:\tN_v = {N_v}\tΔv= {Δv}')

# distribution
μ       = np.zeros((N_t+1, N_x+1, N_v+1))
A       = (x_end-x_start+Δx) * (v_end-v_start+Δv)  # Area
intFac  = A / ((N_x+1) * (N_v+1))
# to calculate a double integral ∫∫(⋅)dxdv we can sum over the grid and multiply by intFac

# grid
X, V = np.meshgrid(x, v, indexing='ij')
xv = np.stack((X, V), axis=-1)  # grid of (x, v) pairs: xv[i, j] = [x[i], v[j]]

def fill_μ_1_2_square(μ, x, v):
    '''Fills μ[0] with initial values μ(t[0]) ~ 𝒰([1,2]×[1,2])'''
    for i in range(len(x)-1):
        for j in range(len(v)-1):
            facx = max(0, min(1, 0.5 + (0.5-abs(x[i]-1.5))/Δx))
            facv = max(0, min(1, 0.5 + (0.5-abs(v[j]-1.5))/Δv))
            μ[0, i, j] = facx * facv

fill_μ_1_2_square(μ, x, v)
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

def mass_MF(μ_t):
    return np.sum(μ_t) * intFac

def max_MF(μ_t):
    flat_index = np.argmax(μ_t)
    ij_max = np.unravel_index(flat_index, μ_t.shape)
    return ij_max / np.array([N_x, N_v]) * np.array([x_end-x_start, v_end-v_start])

def mean_MF(μ_t):
    return np.sum(xv * μ_t[..., np.newaxis], axis=(0, 1)) * intFac / mass_MF(μ_t)

def variance_MF_1d(μ_t, of='v', ):
    """
    calculate x or v variance of μ or μ_t
    of: 'x' or 'v'
    """
    if of == 'x':
        domainVec = x
        intFac1d = (x_end - x_start + Δx) / (N_x + 1)
        μ_flat = μ_t.sum(axis=-1)
    elif of == 'v':
        domainVec = v
        intFac1d = (v_end - v_start + Δv) / (N_v + 1)
        μ_flat = μ_t.sum(axis=-2)
    else:
        raise ValueError("parameter 'of' must be 'x' or 'v'")
    mean_1d = np.sum(domainVec * μ_flat[..., :], axis=-1) * intFac / mass_MF(μ_flat)
    diff = x - mean_1d
    norm_result = np.abs(diff)
    integral = norm_result**2 * μ_flat
    return integral.sum() * intFac# / mass_MF(μ_flat)


def variance_MF_2d(μ_t):
    mean_μ_t = mean_MF(μ_t)
    integral = np.linalg.norm(xv - mean_μ_t[np.newaxis, np.newaxis, :], axis=(-1))**2 * μ_t
    return integral.sum() * intFac


# plotting μ as 2d heatmap:
def plot(μ, t_index):
    plt.imshow(μ[t_index, :, :], extent=[v_start-Δv/2, v_end+Δv/2, x_start-Δx/2, x_end+Δx/2], aspect='auto', origin='lower')

    print(max_MF(μ[t_index]).shape)
    max_μ   = (max_MF(μ[t_index]))[::-1]  # [::-1] flips (x,v) to (v,x)
    mean_μ = (mean_MF(μ[t_index]))[::-1]
    var_μ_x = variance_MF_1d(μ[t_index], 'x')
    var_μ_v = variance_MF_1d(μ[t_index], 'v')
    print(f'\nt={t[t_index]:.3f}\tmax at (v,x) =\t{max_μ}\n\tmean at (v,x) =\t{mean_μ}\n\tvariance x =\t{var_μ_x}\n\tvariance v =\t{var_μ_v}\n\tmass =\t{np.sum(μ[0])*intFac}')
    plt.plot(*max_μ, 'kx', fillstyle='none', label='max')
    plt.plot(*mean_μ, 'r+', fillstyle='none', label='mean')
    var_ellipse  =  Ellipse(xy=mean_μ, width=var_μ_v, height=var_μ_x, color='r', fill=False, label='variance')
    #plt.gca().add_patch(var_ellipse)

    plt.xlabel('velocity $v$')
    plt.ylabel('position $x$')
    plt.title(f'$\\mu(t,x,v)$, $t[{t_index}]={t[t_index]:.3f}$, mass: {np.sum(μ[0])*intFac:.3f}')
    #plt.colorbar(label='$\\mu$ value')
    plt.gca().set_aspect('equal')
    #plt.legend()    
    plt.show()


########## Solving positions (x) and velocities (v) ##########
# functions to compute  h_(i,j) = ∫∫ H_β(y-x[i]) * (w-v[j]) * μ(n,y,w) dy dw
# all three functions compute the same result, but optimized for different problem sizes
def h_LoopLoop():  # for large dimensions (N_x, N_v)
    h_H_β = H_β_y_x_i[:,:,np.newaxis] * μ[np.newaxis, n, :, :]
    for i in range(N_x+1):
        for j in range(0, N_v+1):
            h_ij[i, j] = (h_H_β[i] * w_v_j[j]).sum(axis=(-2, -1))
    h = h_ij[:, :] * μ[n, :, :]
    return h

def h_LoopArr():  # for medium dimensions (N_x, N_v)
    H_β_ij = H_β(y_x)[:,np.newaxis] * w_v
    Integrand = np.lib.stride_tricks.sliding_window_view(H_β_ij, (N_x+1, N_v+1))
    Integral = np.zeros((N_x+1, N_v+1))
    for i in range(N_x+1):
        Integral[i,:] = (Integrand[i,:] * μ[n, np.newaxis, :, :]).sum(axis=(-1, -2))
    h = Integral[::-1, ::-1] * μ[n]
    return h

def h_ArrArr():  # for small dimensions (N_x, N_v)
    H_β_ij = H_β(y_x)[:,np.newaxis] * w_v
    Integrand = np.lib.stride_tricks.sliding_window_view(H_β_ij, (N_x+1, N_v+1))
    Integral = (Integrand * μ[n, np.newaxis, np.newaxis, :, :]).sum(axis=(-1, -2))
    h = Integral[::-1, ::-1] * μ[n]
    return h

# choose h function based on problem size
if N_x * N_v <= 400:
    hfunc = h_ArrArr
elif N_x * N_v <= 2500:
    hfunc = h_LoopArr
else:
    # precompute arrays for h_(i,j) calculation:
    h_ij = np.zeros((N_x+1, N_v+1))
    H_β_y_x_i = H_β(np.lib.stride_tricks.sliding_window_view(y_x, N_x+1))[::-1] # H_β_y_x_i[i] = [H_β(y-x[i]) for all y in position-domain]
    w_v_j = np.lib.stride_tricks.sliding_window_view(w_v, N_v+1)[::-1] # w_v_j[j] = [(w-v[j]) for all w in velocity-domain]
    hfunc = h_LoopLoop
print(f'Function used for calculating h:\t{hfunc.__name__}')

def k_γ_doubleIntegral_v(μ_t, ν_t, γ=1.0/np.sqrt(2)):
    """
    Double-integral kernel k(μ, ν) = ∫∫ k(x, x') dμ(x) dν(x')
    """
    assert μ_t.shape == ν_t.shape
    result = 0.0
    for i in range(N_x+1):
        for j in range(N_v+1):
            w_v_j = np.lib.stride_tricks.sliding_window_view(w_v, N_v+1)[::-1] # w_v_j[j] = [(w-v[j]) for all w in velocity-domain]
            k_w_v = shared_functions.k_γ(w_v_j, γ) # k_w_v[j] = [k_γ(w, v[j]) for all w in velocity-domain]
            innerDoubleInt = k_w_v[np.newaxis, j, :] * μ_t[:, j, np.newaxis]
            #print(k_w_v[np.newaxis, j, :].shape, μ_t[:, j, np.newaxis].shape, innerDoubleInt.shape)
            result += innerDoubleInt.sum() * ν_t[i, j]
    return result * (intFac ** 2)

#start_time = time.time()
for n in range(N_t):
    print(f"\tsolving time step:\t{str(n+1).rjust(len(str(N_t-1)))} / {N_t-1}\t({(n+1)/(N_t-1):.0%})\tmass: {np.sum(μ[0])*intFac}", end="\r")

    if n == 0 or n == 100:
        plot(μ, n)
    
    
    μn_LF = (μ[n, 0:-2, 1:-1] + μ[n, 2:, 1:-1] + μ[n, 1:-1, 0:-2] + μ[n, 1:-1, 2:]) / 4 # Lax-Friedrich
    g = v[np.newaxis, 1:-1] * (μ[n, 2:, 1:-1] - μ[n, 0:-2, 1:-1]) # g_i+1,j - g_i-1,j
    h = intFac * hfunc() # h_(i,j)
    μ[n+1, 1:-1, 1:-1] = μn_LF - (Δt/(2*Δx))*g - (Δt/(2*Δv))*(h[1:-1, 2:] - h[1:-1, :-2])

    # boundary conditions: 0 outwards normal derivative
    μ[n+1,  0 ,  : ] = μ[n+1,   1  ,  :  ].copy()
    μ[n+1, N_x,  : ] = μ[n+1, N_x-1,  :  ].copy()
    μ[n+1,  : ,  0 ] = μ[n+1,   :  ,  1  ].copy()
    μ[n+1,  : , N_v] = μ[n+1,   :  ,N_v-1].copy()

plot(μ, N_t)
#end_time = time.time()
#print(f'\ntook {end_time - start_time:.2f} seconds')


varv = np.array([variance_MF_1d(μ[i], 'v') for i in range(N_t+1)])
varx = np.array([variance_MF_1d(μ[i], 'x') for i in range(N_t+1)])
varB = np.array([variance_MF_2d(μ[i]) for i in range(N_t+1)]) # variance in both dimensions


# interpolation of varv
samples_indices = ((N_t)//(s-1)) * np.arange(0, s, 1)
t_samples = t[samples_indices]
y = varv[samples_indices]
varv_int = shared_functions.interpolate(μ, samples_indices, y, lambda μ, ν: k_γ_doubleIntegral_v(μ, ν, γ))

plt.plot(t, varv, label="$\\mathcal{V}(\\mu^v)$")
plt.plot(t, varx, label="$\\mathcal{V}(\\mu^x)$")
#plt.plot(t, varB, label="$\\mathcal{V}(\\mu)$")
plt.plot(t, varv_int, 'r--', label="interpolated $\\mathcal{V}(\\mu^v)$")
plt.plot(t_samples, y, marker='o', markeredgecolor='orange', fillstyle='none', linestyle=' ', label="known data points")
plt.ylim(0, None)  # set y-axis bottom to 0
plt.title("Variance")
plt.xlabel("$t$")
plt.legend()
plt.show()


########## Interpolation error ##########
err = np.abs(varv - varv_int)

# Plotting:
plt.semilogy(t, err, '.', label="error")
plt.plot(t_samples, err[samples_indices], marker='o', markeredgecolor='r', fillstyle='none', linestyle=' ', label="known data points")
plt.gca().set_xlim(t_start, t_end)
plt.title("Error")
plt.legend()
plt.show()
