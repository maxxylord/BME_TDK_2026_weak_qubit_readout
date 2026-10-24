import numpy as np
import matplotlib.pyplot as plt
fidmap=np.load('fidelity_avg_map.npy')
Nlist=np.linspace(100, 2000, 10, dtype=int)
plt.scatter(Nlist, fidmap)
plt.title("Averaging inference fidelity for p=0.505; q=1.23744*10**(-7)")
plt.xlabel("N number of measurements")
plt.ylabel("Fidelity")
plt.grid()
plt.show()