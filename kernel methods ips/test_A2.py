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

# interaction function
def H_β(x_i, x_j):
    return 1 / (1 + np.linalg.norm(x_i - x_j)**2)**β

# SE-kernel (squared exponential)
def k_γ(x, xʹ):
    # As part of k_γ we have to calculate the 2-norm  ||x − xʹ||
    # This model is 1d, and np.linalg.norm doesn't work on scalars, so instead we use np.abs
    return np.exp(np.abs(x - xʹ)**2 / (-2.0 * γ**2))

# interpolation parameter
s = 5       # number of time samples


########## Calculations ##########


########## Plotting ##########