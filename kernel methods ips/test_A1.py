import numpy as np
import matplotlib.pyplot as plt
import shared_functions


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
v   = np.zeros((N,M))           # velocities
print(f'Number of agents:\tM = {M}')

# initial values
x[0,:] = np.random.rand(M) + np.ones((M))   # random positions  in interval [1,2]
v[0,:] = np.random.rand(M) + np.ones((M))   # random velocities in interval [1,2]

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


########## Solving positions (x) and velocities (v) ##########
# iterative solver (slower)
'''for n in range(N-1):
    print(f"\tsolving time step:\t{str(n+1).rjust(len(str(N-1)))} / {N-1}\t({(n+1)/(N-1):.0%})", end="\r")
    # solving x
    x[n+1,:] = x[n,:] + Δt*v[n,:]
    # solving v
    for i in range(M):
        sum = 0.0
        for j in range(M):
            sum += H_β(x[n,i] - x[n,j]) * (v[n,j] - v[n,i])
        v[n+1,i] = v[n,i] + (Δt/M)*sum 
print()'''

# numpy solver (faster)
for n in range(N-1):
    print(f"\tsolving time step:\t{str(n+1).rjust(len(str(N-1)))} / {N-1}\t({(n+1)/(N-1):.0%})", end="\r")
    # solving x
    x[n+1,:] = x[n,:] + Δt*v[n,:]       # x[n+1,:] shape: (M,)
    # solving v
    diffx = x[n,:,np.newaxis] - x[n,:]  # diffx[i,j] = x_i-x_j    diffx shape: (M, M)
    diffv = v[n,:] - v[n,:,np.newaxis]  # diffv[i,j] = v_j-v_i    diffv shape: (M, M)
    v[n+1,:] = v[n,:] + (Δt/M) * np.sum(H_β(diffx, β) * diffv, 1)     # v[n+1] shape: (M,)
print()


# Plotting positions (x) over time (t)
plt.plot(t, x)
plt.gca().set_xlim(t_0, T)  # set x-axis to interval [t_0, T]
plt.title("Positions")
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.show()

# Plotting velocities (v) over time (t)
plt.plot(t, v)
plt.gca().set_xlim(t_0, T)  # set x-axis to interval [t_0, T]
plt.title("Velocities")
plt.xlabel("$t$")
plt.ylabel("$v$")
plt.show()


########## Variance of v and interpolation ##########
v_var = v.var(axis=1)

# interpolation of v_var
t_samples_indices = ((N-1)//(s-1)) * np.arange(0, s, 1)
t_samples = t[t_samples_indices]
y = v_var[t_samples_indices]
v_var_int = shared_functions.interpolate(t, t_samples_indices, y, lambda x, xʹ: shared_functions.k_γ(x, xʹ, γ))

# Plotting Variance of velocities (v_var) and interpolation (v_var_int) over time (t)
plt.plot(t, v_var, label="true variance $\\mathcal{V}_M$")
plt.plot(t, v_var_int, 'r--', label="interp. variance $\\mathcal{\\hat{V}}_M$")
plt.plot(t_samples, y, marker='o', markeredgecolor='orange', fillstyle='none', linestyle=' ', label="known data points")
plt.gca().set_xlim(t_0, T)  # set x-axis to interval [t_0, T]
plt.title("Velocities variance")
plt.xlabel("$t$")
plt.legend()
plt.show()


########## Interpolation error ##########
err = np.abs(v_var - v_var_int)

# Plotting:
plt.semilogy(t, err, '.', label="error")
'''locations, labels = plt.xticks()
plt.xticks(t_samples, minor=False)
plt.grid(True, which='major', axis='x')
plt.xticks(locations, labels=locations, minor=True)'''
plt.plot(t_samples, err[t_samples_indices], marker='o', markeredgecolor='r', fillstyle='none', linestyle=' ', label="known data points")
plt.gca().set_xlim(t_0, T)  # set x-axis to interval [t_0, T]
plt.title("Error")
plt.legend()
plt.show()