import numpy as np

#N = 1000
#p = 0.505
#q = 1.23744*10**(-7)
p = 0.55
q = 10**(-3)
A = np.array([[(1-p)*(1-q), (1-p)*q],
              [q*p, (1-q)*p]])

B =  np.array([[p*(1-q), q*p],
                   [q*(1-p), (1-q)*(1-p)]])

def calculate(start, N):
    current = np.zeros((N+1, 2))
    current[0] = start

    for k in range(N):
        next = np.zeros((N+1, 2))

        for i in range(k+2):
            if i == 0:
                next[i] = current[i] @ A
            elif i == k+1:
                next[i] = current[i-1] @ B
            else:
                next[i] = current[i] @ A + current[i-1] @ B

        current = next

    return np.array([np.sum(current[i]) for i in range(N+1)])
def fidelity(N):
    left  = calculate(np.array([1, 0]),N)
    right = calculate(np.array([0, 1]),N)
    return 0.5*np.sum([max(left[i], right[i]) for i in range(N+1)])

Nlist=np.linspace(10,500,50).astype(int)
fidmap=np.array([fidelity(N) for N in Nlist])
np.save('fidelity_avg_map_p055.npy', fidmap)