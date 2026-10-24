
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


# ============================================================
# SWEEP AND SIMULATION PARAMETERS
# ============================================================

# Choose "nu" or "dgamma"
SWEEP_PARAMETER = "nu"

epsilon = 10.0       # μeV
nu = 1.0             # μeV
gamma = 5.0          # μeV
dgamma = 0.5         # μeV

hbar = 0.658212      # μeV ns
dt = 0.11            # ns

# Parameter sweep (μeV)
'''
Nparam = 30

if SWEEP_PARAMETER == "nu":
    parameter_values = np.linspace(0.0, 2.0, Nparam)
elif SWEEP_PARAMETER == "dgamma":
    parameter_values = np.linspace(0.0, 2.0, Nparam)
else:
    raise ValueError("SWEEP_PARAMETER must be 'nu' or 'dgamma'.")
'''
# Parameter sweep (μeV)
Nparam = 20

if SWEEP_PARAMETER == "nu":
    parameter_values = np.logspace(np.log10(0.25), np.log10(15.0), Nparam)

elif SWEEP_PARAMETER == "dgamma":
    parameter_values = np.concatenate((
        [0.00000001],
        np.logspace(-5, np.log10(epsilon), Nparam - 1)
    ))

else:
    raise ValueError("SWEEP_PARAMETER must be 'nu' or 'dgamma'.")
# Trajectory simulation
Nmax = 2000
M = 10000
SEED = None

# Saturation convergence
x = 0.005
Ntail_tolerance = x * Nmax
max_iterations = max(1, Nmax // 100)

PROB_TOL = 1e-8


# ============================================================
# BASIS, PROJECTORS AND DETECTOR RESET
# ============================================================

L = np.array([1, 0], dtype=complex)
R = np.array([0, 1], dtype=complex)

sigz = np.array([[1, 0], [0, -1]], dtype=complex)
sigx = np.array([[0, 1], [1, 0]], dtype=complex)

P_l = np.outer(L, L.conj())
P_r = np.outer(R, R.conj())

id2 = np.eye(2, dtype=complex)

# Detector is reset to |B> = |L> before every measurement.
P_B = P_l


# ============================================================
# JOINT UNITARY
# ============================================================

def make_unitary(epsilon, nu, gamma, dgamma, dt):
    H_q = epsilon * sigz + nu * sigx
    H_d = gamma * sigx
    H_int = dgamma * np.kron(P_r, sigx)

    H_tot = (
        np.kron(H_q, id2)
        + np.kron(id2, H_d)
        + H_int
    )

    U = la.expm(-1j * H_tot * dt / hbar)

    # Input system state is tensored with detector |B>.
    # Basis order: |L,B>, |L,T>, |R,B>, |R,T>.
    V = U[:, [0, 2]]

    return U, V


# ============================================================
# PROBABILITY CHECK
# ============================================================

def check_probabilities(pT, pB, context=""):
    err = np.max(np.abs(pT + pB - 1.0))

    if err > PROB_TOL:
        raise RuntimeError(
            f"Probability normalization failed {context}: "
            f"maximum error = {err:.3e}"
        )


# ============================================================
# QUANTUM FULL-TRAJECTORY ML: MONTE CARLO
#
# Returns fidelity for every N = 1,...,Nmax.
# ============================================================

def simulate_all_lengths(Nmax, M, V, seed=None):
    rng = np.random.default_rng(seed)

    # True initial state: 0 = L, 1 = R.
    true_state = rng.integers(0, 2, size=M)

    psi = np.zeros((M, 2), dtype=complex)
    psi[np.arange(M), true_state] = 1.0

    # Two hypotheses per trajectory:
    # hypothesis 0 starts in L; hypothesis 1 starts in R.
    states = np.zeros((M, 2, 2), dtype=complex)
    states[:, 0, 0] = 1.0
    states[:, 1, 1] = 1.0

    logL = np.zeros(M)
    logR = np.zeros(M)

    Results = np.zeros(Nmax)

    rows = np.arange(M)

    for n in range(Nmax):

        # ----------------------------------------------------
        # Generate the next measurement
        # ----------------------------------------------------

        joint = (psi @ V.T).reshape(M, 2, 2)

        pB = np.sum(np.abs(joint[:, :, 0])**2, axis=1)
        pT = np.sum(np.abs(joint[:, :, 1])**2, axis=1)

        check_probabilities(pT, pB, f"generation, step {n + 1}")

        outcome = (rng.random(M) < pT).astype(int)

        unnormalized = joint[rows, :, outcome]
        probs_true = np.where(outcome == 1, pT, pB)
        norm_true = np.sqrt(probs_true)

        psi = unnormalized / norm_true[:, None]

        # ----------------------------------------------------
        # Propagate both inference hypotheses
        # ----------------------------------------------------

        joint_h = (states @ V.T).reshape(M, 2, 2, 2)

        pB_h = np.sum(np.abs(joint_h[:, :, :, 0])**2, axis=2)
        pT_h = np.sum(np.abs(joint_h[:, :, :, 1])**2, axis=2)

        check_probabilities(
            pT_h, pB_h, f"inference, step {n + 1}"
        )

        probs = np.where(
            outcome[:, None] == 1,
            pT_h,
            pB_h
        )

        with np.errstate(divide="ignore"):
            log_probs = np.log(probs)

        logL += log_probs[:, 0]
        logR += log_probs[:, 1]

        # ----------------------------------------------------
        # Normalize conditional states under each hypothesis
        # ----------------------------------------------------

        selected = np.take_along_axis(
            joint_h,
            outcome[:, None, None, None],
            axis=3
        )[:, :, :, 0]

        norms = np.sqrt(probs)
        valid = norms > 0

        states = np.where(
            valid[:, :, None],
            selected / np.where(valid, norms, 1.0)[:, :, None],
            states
        )

        # ----------------------------------------------------
        # ML decision for each trajectory prefix
        # ----------------------------------------------------

        guess = (logR >= logL).astype(int)
        Results[n] = np.mean(guess == true_state)

    return Results


# ============================================================
# EXACT QUANTUM AVERAGING-INFERENCE FIDELITY
#
# Dynamic programming over the number of T outcomes.
# The temporal order of the outcomes is discarded.
# ============================================================

def averaging_fidelity_map(
    Nmax, epsilon, nu, gamma, dgamma, dt
):
    U, _ = make_unitary(
        epsilon, nu, gamma, dgamma, dt
    )
    Udag = U.conj().T

    # current[s, k] is the unnormalized system density matrix
    # for initial hypothesis s and exactly k T outcomes.
    current = np.zeros(
        (2, Nmax + 1, 2, 2),
        dtype=complex
    )

    current[0, 0] = P_l
    current[1, 0] = P_r

    fidmap = np.zeros(Nmax)

    # Joint-basis indices for detector outcomes B and T.
    idx_B = np.array([0, 2])
    idx_T = np.array([1, 3])

    for n in range(Nmax):

        next_ = np.zeros_like(current)

        # There are n measurements before this iteration.
        for s in range(2):
            for k in range(n + 1):

                rho = current[s, k]

                if not np.any(rho):
                    continue

                # Reset detector to |B>, then evolve jointly.
                joint_rho = np.kron(rho, P_B)
                evolved = U @ joint_rho @ Udag

                # B outcome: count remains k.
                rho_B = evolved[np.ix_(idx_B, idx_B)]
                next_[s, k] += rho_B

                # T outcome: count increases to k+1.
                rho_T = evolved[np.ix_(idx_T, idx_T)]
                next_[s, k + 1] += rho_T

        current = next_

        # Probability distributions for the two initial states.
        prob_L = np.trace(
            current[0, :n + 2], axis1=1, axis2=2
        ).real

        prob_R = np.trace(
            current[1, :n + 2], axis1=1, axis2=2
        ).real

        sum_L = prob_L.sum()
        sum_R = prob_R.sum()

        if not np.isclose(sum_L, 1.0, atol=PROB_TOL):
            raise RuntimeError(
                f"Probabilities for L sum to {sum_L} at N={n + 1}."
            )

        if not np.isclose(sum_R, 1.0, atol=PROB_TOL):
            raise RuntimeError(
                f"Probabilities for R sum to {sum_R} at N={n + 1}."
            )

        # Equal prior probabilities; optimal inference from count k.
        fidmap[n] = 0.5 * np.maximum(prob_L, prob_R).sum()

    return fidmap


# ============================================================
# FIND TAIL CUTOFF FROM THE MAX-MIN CRITERION
# ============================================================

def find_tail_cutoff(Results, delta):
    # Range of every suffix Results[i:].
    suffix_max = np.maximum.accumulate(Results[::-1])[::-1]
    suffix_min = np.minimum.accumulate(Results[::-1])[::-1]

    ranges = suffix_max - suffix_min
    eligible = np.flatnonzero(ranges <= delta)

    if len(eligible) == 0:
        return len(Results)

    return int(eligible[0] + 1)


# ============================================================
# ARRAYS FOR RESULTS
# ============================================================

Fsat_values = np.zeros(Nparam)
Favgmax_values = np.zeros(Nparam)

Ntail_values = np.zeros(Nparam, dtype=int)
Navgmax_values = np.zeros(Nparam, dtype=int)

iterations_values = np.zeros(Nparam, dtype=int)
delta_values = np.zeros(Nparam)
converged_values = np.zeros(Nparam, dtype=bool)


# ============================================================
# PARAMETER SWEEP
# ============================================================

for j, value in enumerate(parameter_values):

    if SWEEP_PARAMETER == "nu":
        nu_current = value
        dgamma_current = dgamma
    else:
        nu_current = nu
        dgamma_current = value

    print(
        f"{j + 1}/{Nparam}: "
        f"{SWEEP_PARAMETER} = {value:.6g} μeV"
    )

    # --------------------------------------------------------
    # Build unitary for the current parameter
    # --------------------------------------------------------

    U, V = make_unitary(
        epsilon,
        nu_current,
        gamma,
        dgamma_current,
        dt
    )

    # --------------------------------------------------------
    # Full-trajectory ML Monte Carlo
    # --------------------------------------------------------

    Results = simulate_all_lengths(
        Nmax, M, V,
        seed=None if SEED is None else SEED + j
    )

    # --------------------------------------------------------
    # Iterative saturation estimate
    # --------------------------------------------------------

    Fsat = 0.5
    Ntail = Nmax
    converged = False

    for iteration in range(1, max_iterations + 1):

        delta = 2.0 * np.sqrt(
            max(Fsat * (1.0 - Fsat), 0.0) / M
        )

        Ntail_new = find_tail_cutoff(Results, delta)

        Fsat_new = np.mean(Results[Ntail_new - 1:])

        cutoff_change = abs(Ntail_new - Ntail)

        Ntail = Ntail_new
        Fsat = Fsat_new

        if cutoff_change <= Ntail_tolerance:
            converged = True
            break

    delta = 2.0 * np.sqrt(
        max(Fsat * (1.0 - Fsat), 0.0) / M
    )

    if not converged:
        print(
            f"    WARNING: saturation cutoff did not converge "
            f"within {max_iterations} iterations."
        )

    print(
        f"    Iterations = {iteration}, "
        f"Ntail = {Ntail}, "
        f"Fsat = {Fsat:.8f}, "
        f"delta = {delta:.3e}"
    )

    # --------------------------------------------------------
    # Exact averaging-inference fidelity
    # --------------------------------------------------------

    fidmap = averaging_fidelity_map(
        Nmax,
        epsilon,
        nu_current,
        gamma,
        dgamma_current,
        dt
    )

    Favgind = int(np.argmax(fidmap))
    Favgmax = fidmap[Favgind]

    print(
        f"    Favg,max = {Favgmax:.8f}, "
        f"Navg,max = {Favgind + 1}"
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    Fsat_values[j] = Fsat
    Favgmax_values[j] = Favgmax

    Ntail_values[j] = Ntail
    Navgmax_values[j] = Favgind + 1

    iterations_values[j] = iteration
    delta_values[j] = delta
    converged_values[j] = converged


# ============================================================
# SAVE RESULTS
# ============================================================

prefix = f"Quantum_{SWEEP_PARAMETER}"

np.save(f"{prefix}_parameter_values.npy", parameter_values)
np.save(f"{prefix}_Fsat.npy", Fsat_values)
np.save(f"{prefix}_Favgmax.npy", Favgmax_values)
np.save(f"{prefix}_Ntail.npy", Ntail_values)
np.save(f"{prefix}_Navgmax.npy", Navgmax_values)
np.save(f"{prefix}_iterations.npy", iterations_values)
np.save(f"{prefix}_delta.npy", delta_values)
np.save(f"{prefix}_converged.npy", converged_values)


# ============================================================
# INFIDELITY VS SWEPT PARAMETER
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    parameter_values,
    1.0 - Fsat_values,
    "o-",
    label=r"Full-trajectory ML: $1-\mathcal{F}_{\mathrm{sat}}$"
)

plt.plot(
    parameter_values,
    1.0 - Favgmax_values,
    "o-",
    label=r"Averaging: $1-\mathcal{F}_{\mathrm{avg,max}}$"
)

plt.xlabel(
    r"$\nu$ ($\mu\mathrm{eV}$)"
    if SWEEP_PARAMETER == "nu"
    else r"$\delta\gamma$ ($\mu\mathrm{eV}$)",
    fontsize=15
)
plt.ylabel(r"$1-\mathcal{F}$", fontsize=15)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{prefix}_infidelity.png", dpi=200)
plt.show()


# ============================================================
# OPTIMAL TRAJECTORY LENGTH VS SWEPT PARAMETER
# ============================================================

plt.figure(figsize=(12, 8))
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{braket}",
})
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.xscale("log")
plt.plot(
    parameter_values,
    Ntail_values,
    "o-",
    label=r"$N_{\mathrm{tail}}$ (ML)"
)

plt.plot(
    parameter_values,
    Navgmax_values,
    "o-",
    label=r"$N_{\mathrm{avg,max}}$"
)

plt.xlabel(
    r"$\nu$ ($\mu\mathrm{eV}$)"
    if SWEEP_PARAMETER == "nu"
    else r"$\delta\gamma$ ($\mu\mathrm{eV}$)",
    fontsize=32
)
plt.ylabel(r"$N$", fontsize=32)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{prefix}_optimal_lengths.png", dpi=200)
plt.show()


# ============================================================
# SUMMARY
# ============================================================

print("\nSummary")
print(
    f"{SWEEP_PARAMETER:>10s} "
    f"{'1-Fsat':>12s} "
    f"{'1-Favgmax':>12s} "
    f"{'Ntail':>8s} "
    f"{'Navgmax':>8s} "
    f"{'Converged':>10s}"
)

for j, value in enumerate(parameter_values):
    print(
        f"{value:10.5f} "
        f"{1 - Fsat_values[j]:12.5e} "
        f"{1 - Favgmax_values[j]:12.5e} "
        f"{Ntail_values[j]:8d} "
        f"{Navgmax_values[j]:8d} "
        f"{str(converged_values[j]):>10s}"
    )
