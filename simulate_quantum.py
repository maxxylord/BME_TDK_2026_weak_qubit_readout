import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

L = np.array([1, 0], dtype=complex)  # Also B
R = np.array([0, 1], dtype=complex)  # Also T

sigz = np.array([[1, 0], [0, -1]], dtype=complex)
sigx = np.array([[0, 1], [1, 0]], dtype=complex)

P_l = np.array([[1, 0], [0, 0]], dtype=complex)
P_r = np.array([[0, 0], [0, 1]], dtype=complex)

id = np.eye(2, dtype=complex)

MT = np.kron(id, P_r)
MB = np.kron(id, P_l)

# ============================================================
# PARAMETERS
# ============================================================
epsilon = 10.0
nu = 1.0
gamma = 5.0
dgamma = 0.5

hbar = 0.658212  # μeV·ns
dt = 0.11        # ns

params = [epsilon, nu, gamma, dgamma]

H_q = params[0] * sigz + params[1] * sigx
H_d = params[2] * sigx
H_int = params[3] * np.kron(P_r, sigx)

H_tot = np.kron(H_q, id) + np.kron(id, H_d) + H_int
U = sp.linalg.expm(-1j * H_tot * dt / hbar)

PROB_TOL = 1e-10


# ============================================================
# PROBABILITY UTILITIES
# ============================================================
def probability(psi, projector):
    return float(np.vdot(psi, projector @ psi).real)


def check_probabilities(pT, pB, context=""):
    if not np.isclose(pT + pB, 1.0, rtol=0.0, atol=PROB_TOL):
        raise RuntimeError(
            f"Probabilities do not sum to one {context}: "
            f"P(T)={pT}, P(B)={pB}, sum={pT+pB}"
        )


# ============================================================
# GENERATE ONE TRAJECTORY
# ============================================================
def samplegen(n, start):
    trace = np.zeros(n, dtype=int)
    psi = start.copy()

    for i in range(n):
        # Reset detector to B=L before every interaction
        evolved = U @ np.kron(psi, L)

        pT = probability(evolved, MT)
        pB = probability(evolved, MB)

        check_probabilities(pT, pB, f"in samplegen, step {i}")

        if np.random.rand() < pT:
            newstate = MT @ evolved / np.sqrt(pT)
            trace[i] = 1
            psi = newstate[[1, 3]]
        else:
            newstate = MB @ evolved / np.sqrt(pB)
            trace[i] = 0
            psi = newstate[[0, 2]]

    return trace


# ============================================================
# UPDATE THE TWO HYPOTHESES BY ONE MEASUREMENT
# ============================================================
def inference_step(state1, state2, logL, logR, outcome, step):
    evolved1 = U @ np.kron(state1, L)
    evolved2 = U @ np.kron(state2, L)

    pT1 = probability(evolved1, MT)
    pB1 = probability(evolved1, MB)
    pT2 = probability(evolved2, MT)
    pB2 = probability(evolved2, MB)

    check_probabilities(pT1, pB1, f"hypothesis L, step {step}")
    check_probabilities(pT2, pB2, f"hypothesis R, step {step}")

    if outcome == 1:  # T
        p1, p2 = pT1, pT2
        projector = MT
        indices = [1, 3]
    else:             # B
        p1, p2 = pB1, pB2
        projector = MB
        indices = [0, 2]

    logL += np.log(p1) if p1 > 0 else -np.inf
    logR += np.log(p2) if p2 > 0 else -np.inf

    # Update each conditional state only if its likelihood is nonzero
    if p1 > 0:
        state1 = (projector @ evolved1 / np.sqrt(p1))[indices]

    if p2 > 0:
        state2 = (projector @ evolved2 / np.sqrt(p2))[indices]

    return state1, state2, logL, logR


# ============================================================
# MONTE CARLO FIDELITY FOR EVERY PREFIX
# ============================================================
def simulate_all_lengths(Nmax, M):
    correct = np.zeros(Nmax, dtype=int)

    for _ in range(M):
        # Choose the initial system state with equal prior probabilities
        if np.random.rand() < 0.5:
            start = L
            true_state = 0
        else:
            start = R
            true_state = 1

        # Generate one trajectory of maximum length
        trace = samplegen(Nmax, start)

        # Initialize the two likelihood hypotheses
        state1 = L.copy()
        state2 = R.copy()
        logL = 0.0
        logR = 0.0

        # Evaluate every prefix without regenerating the trajectory
        for n in range(Nmax):
            state1, state2, logL, logR = inference_step(
                state1, state2, logL, logR, trace[n], n
            )

            guess = 0 if logL > logR else 1

            if guess == true_state:
                correct[n] += 1

    return correct / M


# ============================================================
# PARAMETERS AND RESULTS
# ============================================================
Nmax = 500
M = 10000
Samples = 50

Nlist = np.linspace(1, Nmax, Samples).astype(int)

# Evaluate every trajectory prefix from 1 to Nmax
Results_all = simulate_all_lengths(Nmax, M)

# Select the requested lengths
Results = Results_all[Nlist - 1]

np.save("Quantum_results.npy", Results)

plt.plot(np.arange(1, Nmax + 1), Results_all, label="Quantum ML")
plt.xlabel("Trajectory length N")
plt.ylabel("Readout fidelity")
plt.ylim(0, 1)
plt.grid()
plt.legend()
plt.show()

print(" N       Fidelity")
for n, fidelity in zip(Nlist, Results):
    print(f"{n:4d}    {fidelity:.5f}")