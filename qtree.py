import numpy as np
import scipy as sp

L=np.array([1,0])
R=np.array([0,1])
sigz=np.array([[1,0],[0,-1]])
sigx=np.array([[0,1],[1,0]])
P_l=np.array([[1,0],[0,0]]) #Also P_b
P_r=np.array([[0,0],[0,1]]) #Also P_t
id=np.eye(2)
M0=np.kron(id,P_l)
M1=np.kron(id,P_r)

def recover_A(C, B):
    A = np.zeros((2, 2), dtype=np.result_type(C, B))

    for i in range(2):
        for j in range(2):
            block = C[2*i:2*i+2, 2*j:2*j+2]

            # Recover a_ij using the Frobenius inner product
            A[i, j] = np.vdot(B, block) / np.vdot(B, B)

    return A

def calculate(start,N,params,dt):
    H_q=params[0]*sigz+params[1]*sigx
    H_d=params[2]*sigx
    H_int=params[3]*np.kron(P_r,sigx)
    H_tot=np.kron(H_q,id)+np.kron(id,H_d)+H_int
    U=sp.linalg.expm(-1j*H_tot*dt)
    V=U.conj().T
    current= np.zeros((N+1,2,2),dtype=complex)
    current[0]=np.outer(start,start.conj())
    for k in range(N):
        next=np.zeros((N+1,2,2),dtype=complex)
        #evolved=np.zeros((N+1,2,2),dtype=complex)
        for i in range(k+2):
            evolved=U@np.kron(current[i],P_l)@V
            if i==0:
                next[0]=recover_A(M0@evolved@M0,P_l)
            elif i==k+1:
                next[k+1]=recover_A(M1@U@np.kron(current[k],P_l)@V@M1,P_r)
            else:
                evolved1=U@np.kron(current[i-1],P_l)@V
                A=recover_A(M0@evolved@M0,P_l)
                B=recover_A(M1@evolved1@M1,P_r)
                next[i]=A+B

        current=next
    return current
def fidelity(N, params, dt):
    rho_L = calculate(L, N, params, dt)
    rho_R = calculate(R, N, params, dt)

    prob_L = np.trace(rho_L, axis1=1, axis2=2).real
    prob_R = np.trace(rho_R, axis1=1, axis2=2).real

    # Check normalization of both probability distributions
    sum_L = np.sum(prob_L)
    sum_R = np.sum(prob_R)
    #print(f"Sum of probabilities for L: {sum_L}")
    #print(f"Sum of probabilities for R: {sum_R}")

    if not np.isclose(sum_L, 1.0, atol=1e-8):
        raise ValueError(f"Probabilities for L sum to {sum_L}, not 1.")

    if not np.isclose(sum_R, 1.0, atol=1e-8):
        raise ValueError(f"Probabilities for R sum to {sum_R}, not 1.")

    return 0.5 * np.maximum(prob_L, prob_R).sum()
epsilon=10.0
nu=1.0
gamma=5.0
dgamma=0.5
params=[epsilon, nu, gamma, dgamma]
dt=0.11

#result=fidelity(5, params, dt)
#print(result)
Nlist=np.linspace(10,500,50).astype(int)
fidmap=np.array([fidelity(N, params, dt) for N in Nlist])
np.save('fidelity_qtree_map.npy', fidmap)