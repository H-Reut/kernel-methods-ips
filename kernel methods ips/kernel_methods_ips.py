import numpy as np
import matplotlib.pyplot as plt


# time
t_0 =  0
T   = 10
N   = 1000
Δt  = (T-t_0) / N
t  = np.linspace(t_0, T, N)

# agents
M   = 30
x   = np.zeros((N,M))
v   = np.zeros((N,M))

# initial values
x[0,:] = np.random.rand(M) + np.ones((M))
v[0,:] = np.random.rand(M) + np.ones((M))

# interaction function
def Hβ(x_i, x_j):
    # global β
    return 1 / (1 + np.linalg.norm(x_i - x_j)**2) #**β



# solving
for n in range(N-1):
    print(f"step {n:d}", end="\r")
    x[n+1,:] = x[n,:] + Δt*v[n,:]
    for i in range(M):
        sum = 0.0
        for j in range(M):
            Hβ(x[n,i], x[n,j]) * (v[n,j] - v[n,i])
        v[n+1,i] = v[n,i] + (Δt/M)*sum 

print()