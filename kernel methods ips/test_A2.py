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

# model parameter              
γ = 1/np.sqrt(2)

# SE-kernel k_γ(x, xʹ)
#       As part of H_β and k_γ we have to calculate the 2-norm  ||x − xʹ||
#       This model is 1d, and np.linalg.norm() doesn't work on scalars, so instead we use np.abs()
def k_γ(x, xʹ):
    return np.exp(np.abs(x - xʹ)**2 / (-2.0 * γ**2))

# interpolation parameter
s = 5       # number of samples of values for β


########## Calculations ##########
# numpy solver (faster)
def solver(x, v, β):
    # interaction function H_β(x-xʹ)
    #       Instead of implementing H_β(x, xʹ), we implement H_β(diff) which must be called with diff=x-xʹ
    def H_β(diff):
        return 1 / (1 + np.abs(diff)**2)**β

    for n in range(N-1):
        # solving x
        x[n+1,:] = x[n,:] + Δt*v[n,:]
        # solving v
        diffx = x[n,:,np.newaxis] - np.transpose(x[n,:], axes=(0))  # diffx[i,j] = x_i-x_j
        diffv = np.transpose(v[n,:], axes=(0)) - v[n,:,np.newaxis]  # diffv[i,j] = v_j-v_i
        v[n+1] = v[n] + (Δt/M) * np.sum(H_β(diffx) * diffv, 1)
    return x, v

def 𝒥(β):
    _, v_β = solver(x.copy(), v.copy(), β)
    𝒱_β = v_β.var(axis=1)
    return Δt * np.sum(𝒱_β)


β_N = 101           # Number of values for β
β_values = np.linspace(0.0, 5.0, β_N)
𝒥_values = np.zeros((β_N))
for i in range(β_N):
    print(f"\tcalculating 𝒥(β) step:\t{str(i).rjust(len(str(β_N)))} / {β_N}\t({(i)/(β_N):.0%})", end="\r")
    β = β_values[i]
    𝒥_values[i] = 𝒥(β)
print()


# interpolation of 𝒥
β_samples_indices = ((β_N-1)//(s-1)) * np.arange(0, s, 1)
β_samples = β_values[β_samples_indices]
y = 𝒥_values[β_samples_indices]
print(f'\nIndices of time samples:\tt_samples_indices = {β_samples_indices}\nTime samples:\tt_samples = {β_samples}\nVariance of velocities at time samples:\ty={y}')
K = k_γ(β_samples[:,np.newaxis], β_samples)   # Kernel-matrix / Gram-matrix
print(f'Kernel-Matrix:\tK = \n{K}\n\tNow solving y=Kα for α')
α = np.linalg.solve(K, y)
print(f'α = {α}')

# calculating the interpolation function 𝒥ˆ
K = k_γ(β_samples[:,np.newaxis], β_values)   # K[n, i] = k_γ(t_n, t_i), where t_i is a time sample, t_n is arbitrary
𝒥ˆ = α @ K

# error |𝒥_values-𝒥ˆ|
err = np.abs(𝒥_values - 𝒥ˆ)
print(f'Timestep samples:\t{β_samples}\nErrors at samples\t{err[β_samples_indices]}')


########## Plotting ##########
# 𝒥 and approximated 𝒥ˆ over time (t)
plt.plot(β_values, 𝒥_values, label="$\\mathcal{J}$")
plt.plot(β_values, 𝒥ˆ, 'r--', label="$\\mathcal{\\hat{J}}$")
plt.plot(β_samples, y, marker='o', markeredgecolor='orange', fillstyle='none', linestyle=' ', label="known data points")
plt.title("Velocities variance")
plt.xlabel("$\\beta$")
plt.legend()
plt.show()

# error plot
plt.semilogy(β_values, err, '.', label="error")
locations, labels = plt.xticks()
plt.xticks(β_samples, minor=False)
plt.grid(True, which='major', axis='x')
plt.xticks(locations, labels=locations, minor=True)
plt.plot(β_samples, err[β_samples_indices], marker='o', markeredgecolor='r', fillstyle='none', linestyle=' ', label="known data points")
plt.title("Error")
plt.legend()
plt.show()