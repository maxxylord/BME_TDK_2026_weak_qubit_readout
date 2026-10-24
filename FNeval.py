import numpy as np
import matplotlib.pyplot as plt
Nmax = 1000
M = 10000
p = 0.55
q = 10**(-3)
Results=np.load("resFN1.npy")
Nlist = np.arange(1, Nmax + 1)
delta=1/np.sqrt(M)
Fmax=Results[-1]
Fmin=Fmax
Ntail=Nmax
for i in range(Nmax-1,1,-1):
    if Results[i]>Fmax:
        Fmax=Results[i]
    elif Results[i]<Fmin:
        Fmin=Results[i]
    if Fmax-Fmin>delta:
        Ntail=i
        break
Fsat=np.mean(Results[Ntail:])
plt.plot(Nlist, Results)
plt.grid()
plt.xlabel("N")
plt.ylabel("Fidelity")
plt.hlines(Fsat, 0, Nmax, color='red', label='Saturation fidelity')
plt.show()
