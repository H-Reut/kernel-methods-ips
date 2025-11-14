import numpy as np
import matplotlib.pyplot as plt
import shared_functions

# repeatable randomness
seed = np.random.randint(2147483647)
print(f'test_A2.py\t\tseed:\t{seed}')
rng = np.random.default_rng(seed=seed)


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
x[0,:] = rng.uniform(1.0, 2.0, M)   # random positions  in interval [1,2]
v[0,:] = rng.uniform(1.0, 2.0, M)   # random velocities in interval [1,2]

# model parameter
β_N = 101               # Number of different values for β
β_values = np.linspace(0.0, 5.0, β_N)       # values for β
γ = 1.0/np.sqrt(2)      # parameter of k_γ

# interpolation parameter
s = 5       # number of samples of values for β


# Interaction function H_β(x-xʹ) for Cucker-Smale systems
def H_β(diff, β=2.0):
    # As part of H_β we have to calculate the 2-norm  ||x − xʹ||
    # Instead of H_β(x, xʹ), we implement H_β(diff) which must be called with diff=x-xʹ
    # The model is 1d, and np.linalg.norm() doesn't work on scalars, so instead we use np.abs()
    return 1 / (1 + np.abs(diff)**2)**β


########## Solving positions (x) and velocities (v) ##########
def solver_Cucker_Smale(x, v, β):
    for n in range(N-1):
        #print(f"\tsolving time step:\t{str(n+1).rjust(len(str(N-1)))} / {N-1}\t({(n+1)/(N-1):.0%})", end="\r")
        # solving x
        x[n+1,:] = x[n,:] + Δt*v[n,:]       # x[n+1,:] shape: (M,)
        # solving v
        diffx = x[n,:,np.newaxis] - x[n,:]  # diffx[i,j] = x_i-x_j    diffx shape: (M, M)
        diffv = v[n,:] - v[n,:,np.newaxis]  # diffv[i,j] = v_j-v_i    diffv shape: (M, M)
        v[n+1,:] = v[n,:] + (Δt/M) * np.sum(H_β(diffx, β) * diffv, 1)     # v[n+1] shape: (M,)
    #print()
    return x, v


########## J and interpolation ##########
# functional J(β):= ∫ₜ₀ᵀ Var(v(t)) dt  (variance of velocities integrated over time)
def J(β):
    _, v_β = solver_Cucker_Smale(x.copy(), v.copy(), β)
    v_β_var = v_β.var(axis=1)    # variance of velocities for each time step, shape: (N,)
    return Δt * np.sum(v_β_var)


J_values = np.zeros((β_N))
for i in range(β_N):
    print(f"\tcalculating J(β)\tstep:\t{str(i).rjust(len(str(β_N)))} / {β_N}\t({(i)/(β_N):.0%})", end="\r")
    β = β_values[i]
    J_values[i] = J(β)
print()


# interpolation of J
β_samples_indices = ((β_N-1)//(s-1)) * np.arange(0, s, 1)
β_samples = β_values[β_samples_indices]
y = J_values[β_samples_indices]
J_int = shared_functions.interpolate(β_values, β_samples_indices, y, lambda x, xʹ: shared_functions.k_γ(x, xʹ, γ))

# Plotting J and interpolated J_int over time (t)
plt.plot(β_values, J_values, label="$\\mathcal{J}$")
plt.plot(β_values, J_int, 'r--', label="$\\mathcal{\\hat{J}}$")
plt.plot(β_samples, y, marker='o', markeredgecolor='orange', fillstyle='none', linestyle=' ', label="known data points")
#plt.gca().set_ylim(0, None)  # set y-axis bottom to 0
plt.title("Velocities variance")
plt.xlabel("$\\beta$")
plt.legend()
plt.show()


########## Interpolation error ##########
err = np.abs(J_values - J_int)

# Plotting
plt.semilogy(β_values, err, '.', label="error")
'''locations, labels = plt.xticks()
plt.xticks(β_samples, minor=False)
plt.grid(True, which='major', axis='x')
plt.xticks(locations, labels=locations, minor=True)'''
plt.plot(β_samples, err[β_samples_indices], marker='o', markeredgecolor='r', fillstyle='none', linestyle=' ', label="known data points")
plt.title("Error")
plt.legend()
plt.show()