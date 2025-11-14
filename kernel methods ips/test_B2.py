import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import shared_functions
import time
from scipy.stats import skew


# repeatable randomness
seed = 1047260852#np.random.randint(2147483647)
print(f'test_B2.py\t\tseed:\t{seed}')
rng = np.random.default_rng(seed=seed)


########## Parameters ##########
# time
t_0 =    0                      # start time
T   =   10                      # end time
N   = 1000                      # number of time steps
Δt  = (T-t_0) / N               # Δt
t   = np.linspace(t_0, T, N)    # all time steps

# agents
M_values = np.array([   10,  100, 1000,10000,10000])     # number of agents
Mˆ_values= np.array([   10,  100, 1000,  100,  100])     # sample size

# model parameters
s_values = np.array([ 8   , 8   , 8   , 8   , 8   ])     # number of time samples
γ_values = 1/np.sqrt(2)*np.array([1, 1, 1, 1, 1])    # parameters of kernel k_γ
𝜎_values = np.array([ 0.0 , 0.0 , 0.0 , 0.0 , 0.01])     # Add noise to the samples with normal distribution 𝒩(𝜇=0, 𝜎²) (𝜎: standard deviation)
λ_values = np.array([ 0.0 , 0.0 , 0.0 , 0.0 , 0.0001])    # regularization parameter λ for interpolation
spacebins = 100                 # number of bins for space axis (x) in histogram (timebins are timesteps t)
assert len(M_values) == len(Mˆ_values) == len(s_values) == len(γ_values) == len(𝜎_values) == len(λ_values), "Parameter arrays must have the same length"

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


# L_infinity norm of errors of v and skew for different s
L_inf_var = np.zeros(len(s_values))
L_inf_skw = np.zeros(len(s_values))


for i in range(len(s_values)):
    M = int(M_values[i])
    Mˆ= int(Mˆ_values[i])
    s = s_values[i]
    γ = γ_values[i]
    𝜎 = 𝜎_values[i]
    λ = λ_values[i]
    
    # positions and initial values
    print(f'\nParameters:\tM = {M}\n\t\tMˆ = {Mˆ}\n\t\ts = {s}\n\t\tγ = {γ}\n\t\tσ = {𝜎}\n\t\tλ = {λ}')
    x = np.zeros((N, M))
    x[0,:] = rng.uniform(1.0, 2.0, M)   # random positions in interval [1,2]

    # solving
    time_start = time.time()
    for n in range(N-1):
        print(f"\tsolving time step:\t{str(n+1).rjust(len(str(N-1)))} / {N-1}\t({(n+1)/(N-1):.0%})", end="\r")
        if Mˆ == M or Mˆ == 0:
            # solving x
            diffx = x[n,:] - x[n,:,np.newaxis]    # diffx[i,j] = x_j-v_i    diffx shape: (M, M)
            x[n+1] = x[n] + (Δt/M) * np.sum(P(diffx) * diffx, 1)     # x[n+1] shape: (M,)
        else:
            # taking sample of x of size Mˆ
            sample = rng.choice(x[n, :], size=Mˆ, replace=False, shuffle=False)  # shape: (Mˆ,)
            P_x_s = P(x[n,:,np.newaxis] - sample[:])    # P_x_s[i,j] = P(x_i, x_i_j) where x_i is ith agent, x_i_j is jth sampled agent.  shape: (M, Mˆ)
            Pi = 1/Mˆ * np.sum(P_x_s, 1)    # Pi[i] = 1/Mˆ * Σⱼ P(x_i, x_i_j)   shape: (M,)
            Xi = 1/Mˆ * np.sum(P_x_s / Pi[:, np.newaxis] * sample[np.newaxis, :], 1)    # Xi[i] = 1/Mˆ * Σⱼ P(x_i, x_i_j)/Pi[i] * x_i_j   shape: (M,)
    
            # solving x
            #diffx = x[n,:] - x[n,:,np.newaxis]    # diffx[i,j] = x_j-v_i    diffx shape: (M, M)
            x[n+1] = (np.ones((M,)) - Δt*Pi) * x[n] + Δt * Pi * Xi     # x[n+1] shape: (M,)
    time_end = time.time()
    print()
    print(f'Solving time for M={M}:\n\t{time_end - time_start:.2f} seconds')

    # variance and skewness
    x_var = x.var(axis=1)
    x_skw = skewness_from_paper(x) # skew(x, axis=1) # 

    # time samples for interpolation
    samples_indices = ((N-1)//(s-1)) * np.arange(0, s, 1)     # shape: (s,)
    t_samples = t[samples_indices]
    # introducing normal noise to the samples: 𝒩(𝜇=0, 𝜎²=0.0001)
    var_samples = x_var[samples_indices] + rng.normal(0, 𝜎, size=s)    # shape: (s,)
    skw_samples = x_skw[samples_indices] + rng.normal(0, 𝜎, size=s)    # shape: (s,)
    # interpolation of x_var and x_skw
    time1 = time.time()
    print('interpolating variance…')
    x_var_int = shared_functions.interpolate(x, samples_indices, var_samples, lambda x, xʹ: shared_functions.k_γ_doubleSum(x, xʹ, γ), λ, Mˆ)
    print('interpolating skewness…')
    x_skw_int = shared_functions.interpolate(x, samples_indices, skw_samples, lambda x, xʹ: shared_functions.k_γ_doubleSum(x, xʹ, γ), λ, Mˆ)
    time2 = time.time()
    print(f'\tInterpolation time:  {time2 - time1:.2f} seconds')

    # errors
    err_var = np.abs(x_var - x_var_int)
    err_skw = np.abs(x_skw - x_skw_int)
    L_inf_var[i] = np.max(err_var)
    L_inf_skw[i] = np.max(err_skw)
    print(f'L∞ Errors\tvar: {L_inf_var[i]:.4e}\n\t\tskw: {L_inf_skw[i]:.4e}')
    
    ########## Plotting ##########
    fig, axs = plt.subplots(2, 3)

    # Title
    subtitle = f'Agents: $M=${M},   MC-sample of agents: $\\hat{{M}}=${Mˆ},   Timesteps: $N=${N},   Samples of timesteps: $s=${s},   $\\gamma=${γ:.3f},   Regularization: $\\lambda=${λ}'
    if 𝜎:
        # if noisy data was used for interpolation
        dataPointLabel = "noisy data points"
        fig.suptitle(subtitle + f'   Noise std. dev.: $\\sigma=${𝜎}')
    else:
        # if exact data was used for interpolation
        dataPointLabel = "known data points"
        fig.suptitle(subtitle)

    # Plotting positions (x) over time (t)
    axs[0, 0].plot(t, x)
    axs[0, 0].set_xlim(t_0, T)  # set x-axis to interval [t_0, T]
    axs[0, 0].set_ylim(1, 2)    # set y-axis to initial interval [1,2]
    axs[0, 0].set_title("Positions graph")
    axs[0, 0].set_xlabel("$t$")
    axs[0, 0].set_ylabel("$x$")

    # Plotting positions (x) over time (t) as historgam
    hist_x = np.repeat(t, M)
    hist_y = x.flatten()
    axs[1, 0].hist2d(hist_x, hist_y, bins=[N, spacebins], range=[[t_0, T], [1, 2]], density=False)
    axs[1, 0].set_title("Positions histogram")
    axs[1, 0].set_xlabel("$t$")
    axs[1, 0].set_ylabel("$x$")

    # Variance of positions (x_var) and interpolated (x_var_int) over time (t)
    axs[0, 1].plot(t, x_var, label="true variance $v_M$")
    axs[0, 1].plot(t, x_var_int, 'r--', label="interp. variance $\\hat{v}_M$")
    #axs[0, 1].plot(t, x_var_intMC, 'g--', label="interp. variance $\\hat{v}_M$")
    axs[0, 1].plot(t_samples, var_samples, marker='o', markeredgecolor='orange', fillstyle='none', linestyle=' ', label=dataPointLabel)
    axs[0, 1].set_ylim(0, None)  # set y-axis bottom to 0
    axs[0, 1].set_xlim(t_0, T)  # set x-axis to interval [t_0, T]
    axs[0, 1].set_title("Positions variance")
    axs[0, 1].set_xlabel("$t$")
    axs[0, 1].legend()
    
    # Skewness of positions (x_skw) and interpolated (x_skw_int) over time (t)
    axs[0, 2].plot(t, x_skw, label="true skewness $s_M$")
    axs[0, 2].plot(t, x_skw_int, 'r--', label="interp. skewness $\\hat{s}_M$")
    #axs[0, 2].plot(t, x_skw_intMC, 'g--', label="interp. skewness $\\hat{s}_M$")
    axs[0, 2].plot(t_samples, skw_samples, marker='o', markeredgecolor='orange', fillstyle='none', linestyle=' ', label=dataPointLabel)
    axs[0, 2].set_xlim(t_0, T)  # set x-axis to interval [t_0, T]
    axs[0, 2].set_title("Positions skewness")
    axs[0, 2].set_xlabel("$t$")
    axs[0, 2].legend()

    # error plot variance
    axs[1, 1].semilogy(t, err_var, '.', label="error $| v_M - \\hat{v}_M |$")
    axs[1, 1].plot(t_samples, err_var[samples_indices], marker='o', markeredgecolor='r', fillstyle='none', linestyle=' ', label=dataPointLabel)
    axs[1, 1].set_xlim(t_0, T)  # set x-axis to interval [t_0, T]
    axs[1, 1].set_title("Error of variance")
    axs[1, 1].set_xlabel("$t$")
    axs[1, 1].legend()
    
    # error plot skewness
    axs[1, 2].semilogy(t, err_skw, '.', label="error $| s_M - \\hat{s}_M |$")
    axs[1, 2].plot(t_samples, err_skw[samples_indices], marker='o', markeredgecolor='r', fillstyle='none', linestyle=' ', label=dataPointLabel)
    axs[1, 2].set_xlim(t_0, T)  # set x-axis to interval [t_0, T]
    axs[1, 2].set_title("Error of skewness")
    axs[1, 1].set_xlabel("$t$")
    axs[1, 2].legend()

    plt.show()

print(f'\nError table:\n   M   |   Mˆ  |   s   |   γ   |noise SD|   L∞ var   |   L∞ skw   \n-------|-------|-------|-------|--------|------------|------------')
for i in range(len(s_values)):
    λ = λ_values[i]
    print(f'{M_values[i]:>6} {f'|' if M_values[i]!=Mˆ_values[i] else '='}{Mˆ_values[i]:>6} |{s_values[i]:>4}   | {γ_values[i]:.3f} |{f'{𝜎_values[i]:.2e}' if 𝜎_values[i] else '    0   '}| {L_inf_var[i]:.4e} | {L_inf_skw[i]:.4e}')
#np.set_printoptions(linewidth=80)
#print(f'\nError table:\n  noise std dev |regulariz.para|   samples    |   L_inf_var  |   L_inf_skw  \n----------------+--------------+--------------+--------------+---------------\n{np.stack((𝜎_values, λ_values, s_values, L_inf_var, L_inf_skw)).transpose()}\n')
