import numpy as np


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
    return np.exp(np.abs(x - xʹ)**2 / (-2.0 * γ**2))


def k_γ_doubleSum(x, xʹ, γ=1.0/np.sqrt(2)):
    """
    Double-sum squared-exponential kernel k_M(x, x') = 1/M² Σᵢ Σⱼ k_γ(x_i, x'_j)
    - x : shape (..., M)
    - xʹ: shape (..., M)
    """
    assert x.shape[-1] == xʹ.shape[-1], f"Input arrays must have same number of agents M in last dimension, but they are:\n\tx.shape  = {x.shape}\n\txʹ.shape = {xʹ.shape}"
    M = x.shape[-1]
    return (1.0 / M**2) * np.sum(k_γ(x[...,:, np.newaxis], xʹ[...,np.newaxis, :], γ=γ), axis=(-1, -2))


def interpolate(t_full, t_samples_indices, y_samples, k, λ=0.0):
    """
    Performs kernel interpolation using the kernel function k.
    Returns the interpolated values at all time steps t_full.
    - t_full: all time steps
    - t_samples_indices: indices of sampled time steps in t_full
    - y_samples: sampled values at t_full[t_samples_indices]
    - k: kernel function to compute the Kernel/Gram matrix K
    - λ: optional regularization parameter (default 0)
    """
    t_samples = t_full[t_samples_indices]    # shape: (s,)
    s = len(t_samples)
    #print(f'\nIndices of time samples:\tt_samples_indices = {t_samples_indices}\nTime samples:\tt_samples = {t_samples}\nValues at time samples:\ty={y_samples}')
    # Kernel-matrix / Gram-matrix
    # K[n, i] = kernel_function(t_n, t_i), where t_i ∈ t_samples, t_n ∈ t_full
    K = k(t_samples[:,np.newaxis], t_full[np.newaxis,:])   # shape: (s, N)
    #print(f'Now solving y=Kα for α with K=\n{K[:,t_samples_indices]}')
    # coefficients α (shape: (s,))
    α = np.linalg.solve(K[:,t_samples_indices] + s*λ*np.eye(s), y_samples)       # with K[:,t_samples_indices] shape: (s, s)]
    return α @ K    # values of interpolation function with shape: (N,)

def interpolate2(t_full, t_samples_indices, y_samples, k, λ=0.0):
    """
    Performs kernel interpolation using the kernel function k.
    Returns the interpolated values at all time steps t_full.
    - t_full: all time steps
    - t_samples_indices: indices of sampled time steps in t_full
    - y_samples: sampled values at t_full[t_samples_indices]
    - k: kernel function to compute the Kernel/Gram matrix K
    - λ: optional regularization parameter (default 0)
    """
    t_samples = t_full[t_samples_indices]    # shape: (s,)
    s = len(t_samples)
    #print(f'\nIndices of time samples:\tt_samples_indices = {t_samples_indices}\nTime samples:\tt_samples = {t_samples}\nValues at time samples:\ty={y_samples}')
    # Kernel-matrix / Gram-matrix
    # K[n, i] = kernel_function(t_n, t_i), where t_i ∈ t_samples, t_n ∈ t_full
    K = k(t_samples[:,np.newaxis], t_full[np.newaxis,:])   # shape: (s, N)
    #print(f'Now solving y=Kα for α with K=\n{K[:,t_samples_indices]}')
    # coefficients α (shape: (s,))
    α = np.linalg.solve(K[:,t_samples_indices] + s*λ*np.eye(s), y_samples)       # with K[:,t_samples_indices] shape: (s, s)]
    return α @ K    # values of interpolation function with shape: (N,)