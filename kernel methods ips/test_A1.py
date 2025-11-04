import numpy as np
import matplotlib.pyplot as plt

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
β = 2                           
γ = 1/np.sqrt(2)

# interaction function H_β(x-xʹ) and SE-kernel k_γ(x, xʹ)
#       As part of H_β and k_γ we have to calculate the 2-norm  ||x − xʹ||
#       This model is 1d, and np.linalg.norm() doesn't work on scalars, so instead we use np.abs()
#       Also, instead of implementing H_β(x, xʹ), we implement H_β(diff) which must be called with diff=x-xʹ
def H_β(diff):
    return 1 / (1 + np.abs(diff)**2)**β

def k_γ(x, xʹ):
    return np.exp(np.abs(x - xʹ)**2 / (-2.0 * γ**2))

# interpolation parameter
s = 4       # number of time samples


########## Calculations ##########
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
    x[n+1,:] = x[n,:] + Δt*v[n,:]
    # solving v
    diffx = x[n,:,np.newaxis] - x[n,:]  # diffx[i,j] = x_i-x_j
    diffv = v[n,:] - v[n,:,np.newaxis]  # diffv[i,j] = v_j-v_i
    v[n+1] = v[n] + (Δt/M) * np.sum(H_β(diffx) * diffv, 1)
print()

# variance
𝒱 = v.var(axis=1)

# interpolation of 𝒱
t_samples_indices = ((N-1)//(s-1)) * np.arange(0, s, 1)
t_samples = t[t_samples_indices]
y = 𝒱[t_samples_indices]
print(f'\nIndices of time samples:\tt_samples_indices = {t_samples_indices}\nTime samples:\tt_samples = {t_samples}\nVariance of velocities at time samples:\ty={y}')
K = k_γ(t_samples[:,np.newaxis], t_samples)   # Kernel-matrix / Gram-matrix
print(f'Kernel-Matrix:\tK = \n{K}\n\tNow solving y=Kα for α')
α = np.linalg.solve(K, y)
print(f'α = {α}')

# calculating the interpolation function 𝒱ˆ
K = k_γ(t_samples[:,np.newaxis], t)   # K[n, i] = k_γ(t_n, t_i), where t_i is a time sample, t_n is arbitrary
𝒱ˆ = α @ K

# error |𝒱-𝒱ˆ|
err = np.abs(𝒱 - 𝒱ˆ)
print(f'Timestep samples:\t{t_samples}\nErrors at samples\t{err[t_samples_indices]}')


########## Plotting ##########
# positions (x) over time (t)
plt.plot(t, x)
plt.title("Positions")
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.show()

# velocities (v) over time (t)
plt.plot(t, v)
plt.title("Velocities")
plt.xlabel("$t$")
plt.ylabel("$v$")
plt.show()

# true Variance of velocities (𝒱) and approximated (𝒱ˆ) over time (t)
plt.plot(t, 𝒱, label="true variance $\\mathcal{V}_M$")
plt.plot(t, 𝒱ˆ, 'r--', label="approx. variance $\\mathcal{\\hat{V}}_M$")
plt.plot(t_samples, y, marker='o', markeredgecolor='orange', fillstyle='none', linestyle=' ', label="known data points")
plt.title("Velocities variance")
plt.xlabel("$t$")
plt.legend()
plt.show()

# error plot
plt.semilogy(t, err, '.', label="error")
locations, labels = plt.xticks()
plt.xticks(t_samples, minor=False)
plt.grid(True, which='major', axis='x')
plt.xticks(locations, labels=locations, minor=True)
plt.plot(t_samples, err[t_samples_indices], marker='o', markeredgecolor='r', fillstyle='none', linestyle=' ', label="known data points")
plt.title("Error")
plt.legend()
plt.show()