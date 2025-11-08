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

# agents
M = 10000                       # number of agents
Mˆ =  100                       # sample size

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
print(f'Solving time for M={M}:\n\t{time_end - time_start:.2f} seconds')

# variance and skewness
x_var = x.var(axis=1)
x_skw = skewness_from_paper(x)
    
########## Plotting ##########
# Plotting positions (x) over time (t)
plt.plot(t, x)
plt.gca().set_xlim(t_0, T)  # set x-axis to interval [t_0, T]
plt.title("Positions")
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.show()

plt.switch_backend('TkAgg')
plt.get_current_fig_manager().window.state('zoomed')    # fullscreen window
plt.hist2d(np.repeat(t, M), x.flatten(), bins=[N, 1000], density=True)
plt.show()


plt.switch_backend('TkAgg')
fig, axs = plt.subplots(2, 2)
plt.get_current_fig_manager().window.state('zoomed')    # fullscreen window

dataPointLabel = "known data points"
fig.suptitle(f'Samples: $s=${s},   Regularization: $\\lambda=${λ}')

# Variance of positions (x_var) and interpolated (x_var_int) over time (t)
axs[0,0].plot(t, x_var, label="true variance $v_M$")
#axs[0,0].plot(t, x_var_int, 'r--', label="interp. variance $\\hat{v}_M$")
#axs[0,0].plot(t_samples, var_samples, marker='o', markeredgecolor='orange', fillstyle='none', linestyle=' ', label=dataPointLabel)
axs[0,0].set_ylim(0, None)  # set y-axis bottom to 0
axs[0,0].set_xlim(t_0, T)  # set x-axis to interval [t_0, T]
axs[0,0].set_title("Positions variance")
axs[0,0].set_xlabel("$t$")
axs[0,0].legend()
    
# Skewness of positions (x_skw) and interpolated (x_skw_int) over time (t)
axs[0,1].plot(t, x_skw, label="true skewness $s_M$")
#axs[0,1].plot(t, x_skw_int, 'r--', label="interp. skewness $\\hat{s}_M$")
#axs[0,1].plot(t_samples, skw_samples, marker='o', markeredgecolor='orange', fillstyle='none', linestyle=' ', label=dataPointLabel)
axs[0,1].set_xlim(t_0, T)  # set x-axis to interval [t_0, T]
axs[0,1].set_title("Positions skewness")
axs[0,1].set_xlabel("$t$")
axs[0,1].legend()

plt.show()

np.set_printoptions(linewidth=80)
print(f'\nError table:\n  noise std dev |regulariz.para|   samples    |   L_inf_var  |   L_inf_skw  \n----------------+--------------+--------------+--------------+---------------\n{np.stack((𝜎_values, λ_values, s_values, L_inf_var, L_inf_skw)).transpose()}\n')