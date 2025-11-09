import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import shared_functions
import time


########## Parameters ##########
# time
t_0 =    0                      # start time
T   =   10                      # end time
N   = 1000                      # number of time steps
Δt  = (T-t_0) / N               # Δt
t   = np.linspace(t_0, T, N)    # all time steps

# agents
M = 10000                       # number of agents
Mˆ =  100                       # sample size
spacebins = 100                 # number of bins for space axis in histogram
s = 8                           # number of time samples
γ = 10                          # parameters of kernel k_γ

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


    
# positions and initial values
x = np.zeros((N, M))
x[0,:] = np.random.rand(M) + np.ones((M))   # random positions in interval [1,2]


# solving
time_start = time.time()
for n in range(N-1):
    print(f"\tsolving time step:\t{str(n+1).rjust(len(str(N-1)))} / {N-1}\t({(n+1)/(N-1):.0%})", end="\r")
    # taking sample of x of size Mˆ
    sample = np.random.default_rng().choice(
        x[n, :], size=Mˆ, replace=False, shuffle=False)  # shape: (Mˆ,)
    P_x_s = P(x[n,:,np.newaxis] - sample[:])    # P_x_s[i,j] = P(x_i, x_i_j) where x_i is ith agent, x_i_j is jth sampled agent.  shape: (M, Mˆ)
    Pi = 1/Mˆ * np.sum(P_x_s, 1)    # Pi[i] = 1/Mˆ * Σⱼ P(x_i, x_i_j)   shape: (M,)
    Xi = 1/Mˆ * np.sum(P_x_s / Pi[:, np.newaxis] * sample[np.newaxis, :], 1)    # Xi[i] = 1/Mˆ * Σⱼ P(x_i, x_i_j)/Pi[i] * x_i_j   shape: (M,)
    
    # solving x
    #diffx = x[n,:] - x[n,:,np.newaxis]    # diffx[i,j] = x_j-v_i    diffx shape: (M, M)
    x[n+1] = (np.ones((M,)) - Δt*Pi) * x[n] + Δt * Pi * Xi     # x[n+1] shape: (M,)
time_end = time.time()
print()
print(f'Solving time for M={M}, Mˆ={Mˆ}:\n\t{time_end - time_start:.2f} seconds')

# Plotting positions as historgam
hist_x = np.repeat(t, M)
hist_y = x.flatten()
plt.hist2d(hist_x, hist_y, bins=[N, spacebins], density=True)
plt.colorbar(format=mtick.PercentFormatter())
plt.show()

# variance and skewness
x_var = x.var(axis=1)
x_skw = skewness_from_paper(x)


t_samples_indices = ((N-1)//(s-1)) * np.arange(0, s, 1)     # shape: (s,)
t_samples = t[t_samples_indices]
# samples
var_samples = x_var[t_samples_indices]  # shape: (s,)
skw_samples = x_skw[t_samples_indices]  # shape: (s,)
# interpolation of x_var and x_skw
x_var_int = shared_functions.interpolate(t, t_samples_indices, var_samples, lambda x, xʹ: shared_functions.k_γ(x, xʹ, γ))
x_skw_int = shared_functions.interpolate(t, t_samples_indices, skw_samples, lambda x, xʹ: shared_functions.k_γ(x, xʹ, γ))

# errors
err_var = np.abs(x_var - x_var_int)
err_skw = np.abs(x_skw - x_skw_int)

########## Plotting ##########
plt.switch_backend('TkAgg')
fig, axs = plt.subplots(2, 2)
plt.get_current_fig_manager().window.state('zoomed')    # fullscreen window

dataPointLabel = "known data points"

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






# Figuring out which γ gives the least interpolation error
γ_values = [0.1, 0.11, 0.156, 0.22, 0.3125, 0.44, 0.625, 0.88, 1.0, 1.25, 1.8, 2.5, 3.5, 5, 7, 10, 12, 14, 16, 17, 17.5, 18, 18.5, 19, 20, 28.3, 40, 56.6, 80, 100]
errors_var = np.zeros(len(γ_values))
errors_skw = np.zeros(len(γ_values))
for i in range(len(γ_values)):
    γ = γ_values[i]

    # time samples for interpolation
    t_samples_indices = ((N-1)//(s-1)) * np.arange(0, s, 1)     # shape: (s,)
    t_samples = t[t_samples_indices]
    # samples
    var_samples = x_var[t_samples_indices]  # shape: (s,)
    skw_samples = x_skw[t_samples_indices]  # shape: (s,)
    # interpolation of x_var and x_skw
    x_var_int = shared_functions.interpolate(t, t_samples_indices, var_samples, lambda x, xʹ: shared_functions.k_γ(x, xʹ, γ))
    x_skw_int = shared_functions.interpolate(t, t_samples_indices, skw_samples, lambda x, xʹ: shared_functions.k_γ(x, xʹ, γ))

    # errors
    errors_var[i] = np.max(np.abs(x_var - x_var_int))
    errors_skw[i] = np.max(np.abs(x_skw - x_skw_int))
    #print(f'Max errors for γ={γ}\n\tvariance:\t{err_var.max()}\n\tskewness:\t{err_skw.max()}')
plt.loglog(γ_values, errors_var, marker='o', label='variance error')
plt.loglog(γ_values, errors_skw, marker='o', label='skewness error')
plt.gca().set_xlabel("$\\gamma$")
plt.gca().set_title("$||v_M-\\hat{v}_M||_{\\infty}$ for different kernel parameters $\\gamma$")
plt.legend()
plt.show()



np.set_printoptions(linewidth=80)
#print(f'\nError table:\n  noise std dev |regulariz.para|   samples    |   L_inf_var  |   L_inf_skw  \n----------------+--------------+--------------+--------------+---------------\n{np.stack((𝜎_values, λ_values, s_values, L_inf_var, L_inf_skw)).transpose()}\n')