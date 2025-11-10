import numpy as np
import matplotlib.pyplot as plt
import shared_functions
import time

########## Parameters ##########
# time
t_0 =    0                      # start time
T   =   10                      # end time
N   = 1000                      # number of time steps
Δt  = (T-t_0) / N               # Δt
t   = np.linspace(t_0, T, N)    # all time steps
print(f'Time interval:\tt_0 = {t_0}\t\tT = {T}\nTime steps:\tN = {N}')

# agents
M   = 30                        # number of agents
x   = np.zeros((N,M))           # positions
print(f'Number of agents:\tM = {M}')

# initial values
x[0,:] = np.random.rand(M) + np.ones((M))   # random positions in interval [1,2]

# model parameters
γ_values = 1/np.sqrt(2) / np.array([1, 1, 1, 1])    # parameters of kernel k_γ

# interpolation parameter
s_values = np.array([ 2   , 4   , 8   , 4   ])  # number of time samples
𝜎_values = np.array([ 0.0 , 0.0 , 0.0 , 0.01])  # Add noise to the samples with normal distribution 𝒩(𝜇=0, 𝜎²) (𝜎: standard deviation)
λ_values = np.array([ 0.0 , 0.0 , 0.0 , 0.01])  # regularization parameter λ for interpolation


# Interaction function P(x-xʹ) for opinion dynamics model
def P(diff):
    # As part of P we have to calculate the 2-norm  ||x − xʹ||
    # Instead of P(x, xʹ), we implement P(diff) which must be called with diff=x-xʹ
    # The model is 1d, and np.linalg.norm() doesn't work on scalars, so instead we use np.abs()
    return np.abs(diff)**2

# Skewness from the paper
def skewness_from_paper(x):
    # The paper "Recent kernel methods for IPS: first numerical results" defines skewness as:
    #       skew(x) = 1/M * Σᵢ ( ||xᵢ-mean(x)|| / std(x) )^3
    # where std() is standard deviation; mean() is average
    # This is different from the common definition of skewness (which is implemented in scipy.stats.skew)
    # Also, again, the model is 1d, hence np.abs() instead of np.linalg.norm()
    temp = ( np.abs(x - x.mean(axis=1)[:,np.newaxis]) / x.std(axis=1)[:,np.newaxis] )**3
    return 1/M * np.sum(temp ,axis=1)


########## Solving positions (x) ##########
for n in range(N-1):
    print(f"\tsolving time step:\t{str(n+1).rjust(len(str(N-1)))} / {N-1}\t({(n+1)/(N-1):.0%})", end="\r")
    # solving x
    diffx = x[n,:] - x[n,:,np.newaxis]    # diffx[i,j] = x_j-v_i    diffx shape: (M, M)
    x[n+1] = x[n] + (Δt/M) * np.sum(P(diffx) * diffx, 1)     # x[n+1] shape: (M,)
print()

# variance and skewness
x_var = x.var(axis=1)
x_skw = skewness_from_paper(x)

# L_infinity norm of errors of v and skew for different s
L_inf_var = np.zeros(len(s_values))
L_inf_skw = np.zeros(len(s_values))


for i in range(len(s_values)):
    s = s_values[i]
    γ = γ_values[i]
    𝜎 = 𝜎_values[i]
    λ = λ_values[i]

    # time samples for interpolation
    t_samples_indices = ((N-1)//(s-1)) * np.arange(0, s, 1)     # shape: (s,)
    t_samples = t[t_samples_indices]
    x_samples = x[t_samples_indices, :]

    # introducing normal noise to the samples: 𝒩(𝜇=0, 𝜎²=0.0001)
    var_samples = x_var[t_samples_indices] + np.random.normal(0, 𝜎, size=s)    # shape: (s,)
    skw_samples = x_skw[t_samples_indices] + np.random.normal(0, 𝜎, size=s)    # shape: (s,)

    K = shared_functions.k_γ_doubleSum(x[t_samples_indices, np.newaxis], x[np.newaxis, :])
    alpha = np.linalg.solve(K[:,t_samples_indices], var_samples)
    x_var_int = (alpha @ K)

    x_var_int = shared_functions.interpolate(x, t_samples_indices, var_samples, lambda x, xʹ: shared_functions.k_γ_doubleSum(x, xʹ, γ), λ)


    # interpolation of x_var and x_skw
    #x_var_int = shared_functions.interpolate(t, t_samples_indices, var_samples, lambda x, xʹ: shared_functions.k_γ(x, xʹ, γ), λ)
    x_skw_int = shared_functions.interpolate(t, t_samples_indices, skw_samples, lambda x, xʹ: shared_functions.k_γ(x, xʹ, γ), λ)

    # errors
    err_var = np.abs(x_var - x_var_int)
    err_skw = np.abs(x_skw - x_skw_int)
    L_inf_var[i] = np.max(err_var)
    L_inf_skw[i] = np.max(err_skw)
    
    ########## Plotting ##########
    plt.switch_backend('TkAgg')
    fig, axs = plt.subplots(2, 2)
    plt.get_current_fig_manager().window.state('zoomed')    # fullscreen window
    if 𝜎:
        # if noisy data was used for interpolation
        dataPointLabel = "noisy data points"
        fig.suptitle(f'Samples: $s=${s},   Regularization: $\\lambda=${λ},   Noise standard deviation: $\\sigma=${𝜎}')
    else:
        # if exact data was used for interpolation
        dataPointLabel = "known data points"
        fig.suptitle(f'Samples: $s=${s},   Regularization: $\\lambda=${λ}')

    # Variance of positions (x_var) and interpolated (x_var_int) over time (t)
    axs[0,0].plot(t, x_var, label="true variance $v_M$")
    axs[0,0].plot(t, x_var_int, 'r--', label="interp. variance $\\hat{v}_M$")
    axs[0,0].plot(t_samples, var_samples, marker='o', markeredgecolor='orange', fillstyle='none', linestyle=' ', label=dataPointLabel)
    axs[0,0].set_ylim(0, None)  # set y-axis bottom to 0
    axs[0,0].set_xlim(t_0, T)  # set x-axis to interval [t_0, T]
    axs[0,0].set_title("Positions variance")
    axs[0,0].set_xlabel("$t$")
    axs[0,0].legend()
    
    # Skewness of positions (x_skw) and interpolated (x_skw_int) over time (t)
    axs[0,1].plot(t, x_skw, label="true skewness $s_M$")
    axs[0,1].plot(t, x_skw_int, 'r--', label="interp. skewness $\\hat{s}_M$")
    axs[0,1].plot(t_samples, skw_samples, marker='o', markeredgecolor='orange', fillstyle='none', linestyle=' ', label=dataPointLabel)
    axs[0,1].set_xlim(t_0, T)  # set x-axis to interval [t_0, T]
    axs[0,1].set_title("Positions skewness")
    axs[0,1].set_xlabel("$t$")
    axs[0,1].legend()

    # error plot variance
    axs[1,0].semilogy(t, err_var, '.', label="error")
    axs[1,0].plot(t_samples, err_var[t_samples_indices], marker='o', markeredgecolor='r', fillstyle='none', linestyle=' ', label=dataPointLabel)
    axs[1,0].set_xlim(t_0, T)  # set x-axis to interval [t_0, T]
    axs[1,0].set_title("Error for variance")
    axs[1,0].legend()
    #plt.show()

    # error plot skewness
    axs[1,1].semilogy(t, err_skw, '.', label="error")
    axs[1,1].plot(t_samples, err_skw[t_samples_indices], marker='o', markeredgecolor='r', fillstyle='none', linestyle=' ', label=dataPointLabel)
    axs[1,1].set_xlim(t_0, T)  # set x-axis to interval [t_0, T]
    axs[1,1].set_title("Error for skewness")
    axs[1,1].legend()

    plt.show()

np.set_printoptions(linewidth=80)
print(f'\nError table:\n  noise std dev |regulariz.para|   samples    |   L_inf_var  |   L_inf_skw  \n----------------+--------------+--------------+--------------+---------------\n{np.stack((𝜎_values, λ_values, s_values, L_inf_var, L_inf_skw)).transpose()}\n')