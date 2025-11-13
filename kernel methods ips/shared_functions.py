import numpy as np
import time
import scipy.spatial.distance as spDist


#def H_β(diff, β=2.0):
#    """
#    Interaction function H_β(x-xʹ) for Cucker-Smale systems
#    instead of H_β(x, xʹ), we implement H_β(diff) which must be called with diff=x-xʹ
#    The model is 1d, and np.linalg.norm() doesn't work on scalars, so we use np.abs()
#    """
#    return 1 / (1 + np.abs(diff)**2)**β


def k_γ(x, xʹ, γ=1.0/np.sqrt(2)):
    """
    Squared-exponential kernel k_γ(x, x') = exp(-||x-x'||² / (2*γ²))
    The model is 1d, and np.linalg.norm() doesn't work on scalars, so we use np.abs()
    """
    #print("k_γ called with x.shape = ", x.shape, "\tand xʹ.shape = ", xʹ.shape)
    return np.exp(np.abs(x - xʹ)**2 / (-2.0 * γ**2))


def k_γ_doubleSum(x, xʹ, γ=1.0/np.sqrt(2)):
    """
    Double-sum kernel k_M(x, x') = 1/M² Σᵢ Σⱼ k_γ(x_i, x'_j)
    - x : shape (..., M)
    - xʹ: shape (..., M)
    """
    assert x.shape[-1] == xʹ.shape[-1], f"Input arrays must have same number of agents M in last dimension, but they are:\n\tx.shape  = {x.shape}\n\txʹ.shape = {xʹ.shape}"
    M = x.shape[-1]
    return (1.0 / M**2) * np.sum(k_γ(x[...,:, np.newaxis], xʹ[...,np.newaxis, :], γ=γ), axis=(-1, -2))



def interpolate(x_full, x_samples_indices, y_samples, k, Mˆ=0, λ=0.0):
    """
    Performs kernel interpolation using the kernel function k.
    Returns the interpolated values at all time steps t_full.
    - x_full: full x-axis
    - x_samples_indices: indices of sampled time steps in x_full
    - y_samples: sampled values at t_full[t_x_samples_indices]
    - k: kernel function to compute the Kernel/Gram matrix K
    - Mˆ: number of agents to use for interpolation. 
          (Mˆ=0 or Mˆ=M: use all agents)
    - λ: optional regularization parameter (0.0 = no regularization)
    """
    x_samples = x_full[x_samples_indices]
    s = len(x_samples)
    N, M = x_full.shape[0:2]
    assert 0 <= Mˆ <= M, f"Mˆ ({Mˆ}) must be ≥0 and ≤M ({M})"
    if Mˆ == 0:
        Mˆ = M
    agents_sample_indices = np.random.default_rng().choice(M, size=Mˆ, replace=False, shuffle=False)
    #print(f'\nIndices of samples:\tx_samples_indices = {x_samples_indices}\nsamples:\tx_samples = {x_samples}\ny values at samples:\ty={y_samples}')
    # Kernel-matrix / Gram-matrix
    # K[i, j] = kernel_function(x_j, x_i), where x_j ∈ x_full, x_i ∈ x_samples
    K = np.zeros((s, N))
    for i in range(s):
        print(f"\tCalculating K sample:\t{str(i+1).rjust(len(str(s)))} / {s}\t({(i+1)/s:.0%})", end="\r")
        x_samples_T_i = x_samples[i, agents_sample_indices]
        for j in range(N):
            K[i, j] = k(x_full[j, agents_sample_indices], x_samples_T_i)
    print()
    #print(f'Now solving y=Kα for α with K=\n{K[:,x_samples_indices]}')
    # coefficients α with shape: (s,)
    α = np.linalg.solve(K[:,x_samples_indices] + s*λ*np.eye(s), y_samples)       # with K[:,x_samples_indices] shape: (s, s)]
    return α @ K    # values of interpolation function with shape: (N,)

def interpolateNoMC(x_full, x_samples_indices, y_samples, k, Mˆ=0, λ=0.0):
    """
    Performs kernel interpolation using the kernel function k.
    Returns the interpolated values at all time steps t_full.
    - x_full: full x-axis
    - x_samples_indices: indices of sampled time steps in x_full
    - y_samples: sampled values at t_full[t_x_samples_indices]
    - k: kernel function to compute the Kernel/Gram matrix K
    - Mˆ: number of agents to use for interpolation. 
          (Mˆ=0 or Mˆ=M: use all agents)
    - λ: optional regularization parameter (0.0 = no regularization)
    """
    x_samples = x_full[x_samples_indices]
    s = len(x_samples)
    N, M = x_full.shape[0:2]
    '''assert 0 <= Mˆ <= M, f"Mˆ ({Mˆ}) must be ≥0 and ≤M ({M})"
    if Mˆ == 0:
        Mˆ = M'''
    #agents_sample_indices = np.random.default_rng().choice(M, size=Mˆ, replace=False, shuffle=False)
    #print(f'\nIndices of samples:\tx_samples_indices = {x_samples_indices}\nsamples:\tx_samples = {x_samples}\ny values at samples:\ty={y_samples}')
    # Kernel-matrix / Gram-matrix
    # K[i, j] = kernel_function(x_j, x_i), where x_j ∈ x_full, x_i ∈ x_samples
    K = np.zeros((s, N))
    for i in range(s):
        print(f"\tCalculating K sample:\t{str(i+1).rjust(len(str(s)))} / {s}\t({(i+1)/s:.0%})", end="\r")
        x_samples_T_i = x_samples[i, :]
        for j in range(N):
            K[i, j] = k(x_full[j, :], x_samples_T_i)
    print()
    #print(f'Now solving y=Kα for α with K=\n{K[:,x_samples_indices]}')
    # coefficients α with shape: (s,)
    α = np.linalg.solve(K[:,x_samples_indices] + s*λ*np.eye(s), y_samples)       # with K[:,x_samples_indices] shape: (s, s)]
    return α @ K    # values of interpolation function with shape: (N,)