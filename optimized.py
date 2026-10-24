import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# ============================================================
# BASIS AND PROJECTORS
# ============================================================
L = np.array([1, 0], dtype=complex)  # Also B
R = np.array([0, 1], dtype=complex)  # Also T

sigz = np.array([[1, 0], [0, -1]], dtype=complex)
sigx = np.array([[0, 1], [1, 0]], dtype=complex)

P_l = np.array([[1, 0], [0, 0]], dtype=complex)
P_r = np.array([[0, 0], [0, 1]], dtype=complex)

id = np.eye(2, dtype=complex)

# ============================================================
# PARAMETERS
# ============================================================
epsilon = 10.0
nu = 1.0
gamma = 5.0
dgamma = 0.5

hbar = 0.658212  # μeV·ns
dt = 0.11        # ns

H_q = epsilon * sigz + nu * sigx
H_d = gamma * sigx
H_int = dgamma * np.kron(P_r, sigx)

H_tot = np.kron(H_q, id) + np.kron(id, H_d) + H_int
U = sp.linalg.expm(-1j * H_tot * dt / hbar)

# Input states are psi ⊗ L = [a, 0, b, 0].
# V maps the system's two amplitudes to the four joint amplitudes.
V = U[:, [0, 2]]

PROB_TOL = 1e-10


# ============================================================
# PROBABILITY CHECK
# ============================================================
def check_probabilities(pT, pB, context=""):
    err = np.max(np.abs(pT + pB - 1.0))
    if err > PROB_TOL:
        raise RuntimeError(
            f"Probability normalization failed {context}: "
            f"maximum error = {err}"
        )


# ============================================================
# VECTORISED MONTE CARLO SIMULATION
# ============================================================
def simulate_all_lengths(Nmax, M, seed=None):
    rng = np.random.default_rng(seed)

    # True initial state: 0 = L, 1 = R
    true_state = rng.integers(0, 2, size=M)

    # System states for trajectory generation
    psi = np.zeros((M, 2), dtype=complex)
    psi[np.arange(M), true_state] = 1.0

    # Two inference hypotheses for every trajectory:
    # hypothesis 0 starts in L; hypothesis 1 starts in R.
    states = np.zeros((M, 2, 2), dtype=complex)
    states[:, 0, 0] = 1.0
    states[:, 1, 1] = 1.0

    logL = np.zeros(M)
    logR = np.zeros(M)

    correct = np.zeros(Nmax, dtype=np.int64)

    for n in range(Nmax):

        # ----------------------------------------------------
        # Generate the next measurement for all M trajectories
        # ----------------------------------------------------
        joint = (psi @ V.T).reshape(M, 2, 2)
        # Axes: trajectory, system state, detector state
        # Detector state 0 = B; detector state 1 = T

        pB = np.sum(np.abs(joint[:, :, 0])**2, axis=1)
        pT = np.sum(np.abs(joint[:, :, 1])**2, axis=1)

        check_probabilities(pT, pB, f"in generation, step {n}")

        outcome = (rng.random(M) < pT).astype(int)

        # Select the unnormalized conditional system states
        rows = np.arange(M)
        unnormalized = joint[rows, :, outcome]
        norm = np.sqrt(np.where(outcome == 1, pT, pB))

        psi = unnormalized / norm[:, None]

        # ----------------------------------------------------
        # Evaluate both initial-state hypotheses in parallel
        # ----------------------------------------------------
        joint_h = (states @ V.T).reshape(M, 2, 2, 2)
        # Axes: trajectory, hypothesis, system, detector

        pB_h = np.sum(np.abs(joint_h[:, :, :, 0])**2, axis=2)
        pT_h = np.sum(np.abs(joint_h[:, :, :, 1])**2, axis=2)

        check_probabilities(
            pT_h, pB_h, f"inference, step {n}"
        )

        # Select likelihoods corresponding to the observed outcomes
        probs = np.where(outcome[:, None] == 1, pT_h, pB_h)

        with np.errstate(divide="ignore"):
            log_probs = np.log(probs)

        logL += log_probs[:, 0]
        logR += log_probs[:, 1]

        # ----------------------------------------------------
        # Update conditional states under each hypothesis
        # ----------------------------------------------------
        selected = np.take_along_axis(
            joint_h,
            outcome[:, None, None, None],
            axis=3
        )[:, :, :, 0]

        norms = np.sqrt(probs)

        # An impossible outcome gives likelihood zero.
        # Retain that hypothesis's previous state to avoid 0/0;
        # its log-likelihood remains -infinity.
        valid = norms > 0

        states = np.where(
            valid[:, :, None],
            selected / np.where(valid, norms, 1.0)[:, :, None],
            states
        )

        # ----------------------------------------------------
        # ML decision for every trajectory prefix
        # ----------------------------------------------------
        guess = (logR >= logL).astype(int)
        correct[n] = np.count_nonzero(guess == true_state)

    return correct / M


# ============================================================
# PARAMETERS AND RESULTS
# ============================================================
Nmax = 500
M = 10000
Samples = 50

Nlist = np.linspace(1, Nmax, Samples).astype(int)

Results_all = simulate_all_lengths(Nmax, M, seed=None)
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