import numpy as np
import scipy as sp

L = np.array([1, 0])
R = np.array([0, 1])

sigz = np.array([[1, 0], [0, -1]])
sigx = np.array([[0, 1], [1, 0]])

P_l = np.array([[1, 0], [0, 0]])  # Also P_b
P_r = np.array([[0, 0], [0, 1]])  # Also P_t

id = np.eye(2)
M0 = np.kron(id, P_l)
M1 = np.kron(id, P_r)


def recover_A(C, B):
    A = np.zeros((2, 2), dtype=np.result_type(C, B))

    for i in range(2):
        for j in range(2):
            block = C[2*i:2*i+2, 2*j:2*j+2]
            A[i, j] = np.vdot(B, block) / np.vdot(B, B)

    return A


def fidelity_map(Nmax, params, dt):
    H_q = params[0] * sigz + params[1] * sigx
    H_d = params[2] * sigx
    H_int = params[3] * np.kron(P_r, sigx)
    H_tot = np.kron(H_q, id) + np.kron(id, H_d) + H_int

    U = sp.linalg.expm(-1j * H_tot * dt)
    V = U.conj().T

    # First axis: initial state (0 = L, 1 = R)
    current = np.zeros((2, Nmax + 1, 2, 2), dtype=complex)
    current[0, 0] = np.outer(L, L.conj())
    current[1, 0] = np.outer(R, R.conj())

    fidmap = np.zeros(Nmax)

    for k in range(Nmax):
        next_ = np.zeros_like(current)

        for i in range(k + 2):
            # B outcome: transmission count remains unchanged
            if i <= k:
                evolved = U @ np.kron(current[:, i], P_l) @ V

                for s in range(2):
                    next_[s, i] += recover_A(
                        M0 @ evolved[s] @ M0, P_l
                    )

            # T outcome: transmission count increases by one
            if i >= 1:
                evolved = U @ np.kron(current[:, i - 1], P_l) @ V

                for s in range(2):
                    next_[s, i] += recover_A(
                        M1 @ evolved[s] @ M1, P_r
                    )

        current = next_

        # Probability distributions after k+1 measurements
        prob_L = np.trace(
            current[0, :k + 2], axis1=1, axis2=2
        ).real

        prob_R = np.trace(
            current[1, :k + 2], axis1=1, axis2=2
        ).real

        # Classical inference fidelity
        fidmap[k] = 0.5 * np.maximum(prob_L, prob_R).sum()

        # Check normalization
        sum_L = prob_L.sum()
        sum_R = prob_R.sum()

        if not np.isclose(sum_L, 1.0, atol=1e-8):
            raise ValueError(
                f"Probabilities for L sum to {sum_L} at N={k + 1}."
            )

        if not np.isclose(sum_R, 1.0, atol=1e-8):
            raise ValueError(
                f"Probabilities for R sum to {sum_R} at N={k + 1}."
            )

    return fidmap


# Parameters
epsilon = 10.0
nu = 1.0
gamma = 5.0
dgamma = 0.5

params = [epsilon, nu, gamma, dgamma]
dt = 0.11

# Desired sequence lengths
Nlist = np.linspace(10, 1500, 50).astype(int)

# Calculate fidelity for every N from 1 to max(Nlist), in one run
fid_all = fidelity_map(Nlist[-1], params, dt)

# Select the requested sequence lengths
fidmap = fid_all[Nlist - 1]

# Save results
np.save("fidelity_qtree_map.npy", fidmap)