
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


# ============================================================
# PARAMETERS
# ============================================================

epsilon = 10.0       # Qubit energy parameter [μeV]
nu = 1.0             # Qubit tunnelling parameter [μeV]
gamma = 5.0          # Detector coupling [μeV]
dgamma = 0.5         # State-dependent detector coupling [μeV]

hbar = 0.658212      # [μeV ns]
dt = 0.11            # Evolution time per measurement [ns]

Nmax = 3000
M = 10000

# Saturation estimation
x = 0.005             # Required Ntail convergence: 0.5% of Nmax
max_iterations = 100


# ============================================================
# HAMILTONIAN AND MEASUREMENT OPERATORS
# ============================================================

I2 = np.eye(2, dtype=complex)

sigx = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

sigz = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

P_L = np.array([
    [1, 0],
    [0, 0]
], dtype=complex)

P_R = np.array([
    [0, 0],
    [0, 1]
], dtype=complex)

# Qubit and detector Hamiltonians
Hq = epsilon * sigz + nu * sigx
Hd = gamma * sigx

# Interaction Hamiltonian
# Joint basis ordering: |L,B>, |L,T>, |R,B>, |R,T>
H_int = -dgamma * np.kron(P_R, sigx)

H_tot = (
    np.kron(Hq, I2)
    + np.kron(I2, Hd)
    + H_int
)

# Unitary time evolution
U = expm(-1j * H_tot * dt / hbar)

# Detector is reset to |B> before each measurement.
# Initial detector state: |B> = [1, 0].
#
# Kraus operators map the qubit state to its unnormalized
# post-measurement state for detector outcomes B and T.

idx_B = [0, 2]
idx_T = [1, 3]
idx_in = [0, 2]

K_B = U[np.ix_(idx_B, idx_in)]
K_T = U[np.ix_(idx_T, idx_in)]

# Completeness check: K_B† K_B + K_T† K_T = I
completeness = (
    K_B.conj().T @ K_B
    + K_T.conj().T @ K_T
)

assert np.allclose(completeness, I2, atol=1e-10)


# ============================================================
# DIRECT ML: MONTE CARLO F(N)
# ============================================================

def simulate_all_lengths(Nmax, M, K_B, K_T, seed=None):
    """
    Estimate direct-ML fidelity after every prefix N=1,...,Nmax.

    The initial qubit state is chosen uniformly from |L>, |R>.
    The detector is reset to |B> before every measurement.

    Returns
    -------
    Results : ndarray, shape (Nmax,)
        Results[N-1] is the estimated fidelity after N measurements.
    """

    rng = np.random.default_rng(seed)

    # Random true initial states: 0 = L, 1 = R
    start = rng.integers(0, 2, size=M)

    # Actual qubit states, initially basis states
    psi = np.zeros((M, 2), dtype=complex)
    psi[np.arange(M), start] = 1.0

    # Two competing initial-state hypotheses for every trajectory:
    # axis 1: hypothesis L or hypothesis R
    psi_h = np.zeros((M, 2, 2), dtype=complex)
    psi_h[:, 0, 0] = 1.0
    psi_h[:, 1, 1] = 1.0

    # Log likelihoods of the complete observed trajectory
    loglike = np.zeros((M, 2))

    Results = np.zeros(Nmax)

    for t in range(Nmax):

        # ----------------------------------------------------
        # Generate the next actual detector outcome
        # ----------------------------------------------------

        cand_B = psi @ K_B.T
        cand_T = psi @ K_T.T

        prob_B = np.sum(np.abs(cand_B)**2, axis=1)
        prob_T = np.sum(np.abs(cand_T)**2, axis=1)

        prob_total = prob_B + prob_T

        # Numerical normalization
        prob_T = prob_T / prob_total

        # outcome = 0 means B; outcome = 1 means T
        outcome = (rng.random(M) < prob_T)

        psi_next = np.where(
            outcome[:, None],
            cand_T,
            cand_B
        )

        norm = np.linalg.norm(psi_next, axis=1)

        psi = psi_next / norm[:, None]

        # ----------------------------------------------------
        # Update both initial-state hypotheses
        # ----------------------------------------------------

        hyp_B = np.einsum("ij,mhj->mhi", K_B, psi_h)
        hyp_T = np.einsum("ij,mhj->mhi", K_T, psi_h)

        chosen = np.where(
            outcome[:, None, None],
            hyp_T,
            hyp_B
        )

        # Conditional outcome probabilities under each hypothesis
        norm2 = np.sum(np.abs(chosen)**2, axis=2)

        # Add log probabilities to obtain trajectory log-likelihoods
        log_norm2 = np.full_like(norm2, -np.inf)
        np.log(norm2, out=log_norm2, where=norm2 > 0)

        loglike = np.where(
            norm2 > 0,
            loglike + log_norm2,
            -np.inf
        )

        # Normalize the hypothesis states for the next iteration.
        # If a hypothesis assigns zero probability to the observed
        # outcome, retain its old state; its likelihood stays -inf.
        valid = norm2 > 0
        denominator = np.sqrt(np.where(valid, norm2, 1.0))

        psi_h = np.where(
            valid[:, :, None],
            chosen / denominator[:, :, None],
            psi_h
        )

        # ----------------------------------------------------
        # Maximum-likelihood estimate of the initial state
        # ----------------------------------------------------

        guess = (loglike[:, 1] > loglike[:, 0]).astype(int)

        Results[t] = np.mean(guess == start)

    return Results


# ============================================================
# AVERAGING ML: EXACT DYNAMIC PROGRAMMING
# ============================================================

def averaging_fidelity(Nmax, K_B, K_T):
    """
    Exact averaging-ML fidelity after every N=1,...,Nmax.

    Trajectories are grouped by the number of T outcomes.
    Their order is discarded.

    Returns
    -------
    fidelities : ndarray, shape (Nmax,)
    """

    # current[s, k] is the unnormalized qubit density matrix
    # for initial state s and exactly k T outcomes.
    #
    # s=0: initial L
    # s=1: initial R

    current = np.zeros(
        (2, Nmax + 1, 2, 2),
        dtype=complex
    )

    current[0, 0] = P_L
    current[1, 0] = P_R

    fidelities = np.zeros(Nmax)

    K_B_dag = K_B.conj().T
    K_T_dag = K_T.conj().T

    for n in range(1, Nmax + 1):

        next_current = np.zeros_like(current)

        # Before this measurement, at most n-1 T outcomes occurred.
        for k in range(n):

            for s in range(2):
                rho = current[s, k]

                # Outcome B: number of T outcomes unchanged
                next_current[s, k] += (
                    K_B @ rho @ K_B_dag
                )

                # Outcome T: number of T outcomes increases by one
                next_current[s, k + 1] += (
                    K_T @ rho @ K_T_dag
                )

        current = next_current

        # Probability of each count, conditional on each initial state
        prob_L = np.real(
            np.trace(current[0, :n + 1], axis1=-2, axis2=-1)
        )

        prob_R = np.real(
            np.trace(current[1, :n + 1], axis1=-2, axis2=-1)
        )

        # Equal prior probabilities for initial L and R:
        # F_avg(N) = 1/2 sum_k max(P(k|L), P(k|R))
        fidelities[n - 1] = 0.5 * np.sum(
            np.maximum(prob_L, prob_R)
        )

    return fidelities


# ============================================================
# SATURATION ESTIMATION
# ============================================================

def estimate_saturation(Results, M, x=0.005, max_iterations=100):
    """
    Iteratively estimate the saturation fidelity and tail cutoff.

    Ntail is the first measurement number in the accepted suffix.
    """

    Nmax = len(Results)

    Fsat = Results[-1]
    Ntail_prev = None

    for iteration in range(1, max_iterations + 1):

        delta = 2 * np.sqrt(Fsat * (1 - Fsat) / M)

        # Suffix extrema:
        # suffix_max[i] = max(Results[i:])
        # suffix_min[i] = min(Results[i:])
        suffix_max = np.maximum.accumulate(Results[::-1])[::-1]
        suffix_min = np.minimum.accumulate(Results[::-1])[::-1]

        suffix_range = suffix_max - suffix_min

        # Earliest suffix whose full range is within the tolerance
        valid = np.flatnonzero(suffix_range <= delta)

        if len(valid) == 0:
            Ntail = Nmax
        else:
            Ntail = int(valid[0] + 1)  # Convert index to N

        # Results[Ntail-1:] corresponds to N >= Ntail
        Fsat_new = np.mean(Results[Ntail - 1:])

        print(
            f"Iteration {iteration}: "
            f"Ntail = {Ntail}, "
            f"Fsat = {Fsat_new:.8f}, "
            f"delta = {delta:.3e}"
        )

        converged = (
            Ntail_prev is not None
            and abs(Ntail - Ntail_prev) <= x * Nmax
        )

        Fsat = Fsat_new

        if converged:
            break

        Ntail_prev = Ntail

    else:
        print("Warning: saturation estimate did not fully converge.")

    delta = 2 * np.sqrt(Fsat * (1 - Fsat) / M)

    print(
        f"\nFinal estimate: Ntail = {Ntail}, "
        f"Fsat = {Fsat:.8f}, delta = {delta:.3e}"
    )

    return Fsat, Ntail, delta


# ============================================================
# COMPUTE F(N)
# ============================================================

print("Running direct-ML Monte Carlo simulation...")
Results = simulate_all_lengths(
    Nmax=Nmax,
    M=M,
    K_B=K_B,
    K_T=K_T,
    seed=42
)

print("\nCalculating exact averaging-ML fidelity...")
fidmap = averaging_fidelity(
    Nmax=Nmax,
    K_B=K_B,
    K_T=K_T
)

Nlist = np.arange(1, Nmax + 1)

Favgmax = np.max(fidmap)
Favgind = int(np.argmax(fidmap))
Navgmax = Favgind + 1

Fsat, Ntail, delta = estimate_saturation(
    Results,
    M,
    x=x,
    max_iterations=max_iterations
)

print(f"\nBest averaging fidelity: {Favgmax:.8f}")
print(f"Optimal averaging sequence length: {Navgmax}")


# ============================================================
# PLOT
# ============================================================

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{braket}"
})

fig, ax = plt.subplots(figsize=(12, 8))

# Avoid zero values on the logarithmic y-axis
tiny = np.finfo(float).tiny

ax.set_yscale("log")
ax.tick_params(axis="both", labelsize=22)

ax.plot(
    Nlist,
    np.maximum(1 - Results, tiny),
    label="Direct ML"
)

ax.plot(
    Nlist,
    np.maximum(1 - fidmap, tiny),
    color="red",
    label="Averaging ML"
)

ax.scatter(
    Navgmax,
    max(1 - Favgmax, tiny),
    color="purple",
    s=100,
    zorder=5,
    label=(
        r"$1-\mathcal{F}_{\mathrm{max}}^{\mathrm{avg}}=$"
        + f"{1 - Favgmax:.3E}"
    )
)

ax.axhline(
    1 - Fsat,
    color="orange",
    linestyle="dashed",
    label=(
        r"$1-\mathcal{F}_{\mathrm{sat}}^{\mathrm{d}}=$"
        + f"{1 - Fsat:.3E}"
    )
)

ax.axhline(
    max(1 - Fsat + delta, tiny),
    color="green",
    linestyle="dashed",
    label=r"$1-\mathcal{F}_{\mathrm{sat}}^{\mathrm{d}}+\sigma^{(1)}$"
)

ax.axhline(
    max(1 - Fsat - delta, tiny),
    color="green",
    linestyle="dotted",
    label=r"$1-\mathcal{F}_{\mathrm{sat}}^{\mathrm{d}}-\sigma^{(1)}$"
)
ax.axvline(
    Ntail,
    color="gray",
    linestyle=":",
    label=rf"$N_{{\mathrm{{cut}}}}={Ntail}$"
)

ax.set_xlabel(r" $N$", fontsize=32)
ax.set_ylabel(r" $1-\mathcal{F}$", fontsize=32)

ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=18)

fig.tight_layout()
plt.show()
