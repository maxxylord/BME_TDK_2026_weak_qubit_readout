import numpy as np
import matplotlib.pyplot as plt
fidmap=np.load('fidelity_qtree_map.npy')
Nlist=np.linspace(10,1500,50).astype(int)
plt.scatter(Nlist, fidmap)
plt.title("Averaging inference fidelity for p=0.505; q=1.23744*10**(-7)")
plt.xlabel("N number of measurements")
plt.ylabel("Fidelity")
plt.grid()
plt.show()