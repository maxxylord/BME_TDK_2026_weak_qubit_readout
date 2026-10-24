import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


# ============================================================
# SWEEP AND SIMULATION PARAMETERS
# ============================================================

SWEEP_PARAMETER = "dgamma"  # "nu" or "dgamma"

epsilon = 10.0       # μeV
nu = 1.0             # μeV
gamma = 5.0          # μeV
dgamma = 0.5         # μeV

hbar = 0.658212      # μeV ns
dt = 0.11            # ns

Nparam = 20

if SWEEP_PARAMETER == "nu":
    # Retains the range used in the supplied script.
    parameter_values = np.linspace(0.25, 15.0, Nparam)
elif SWEEP_PARAMETER == "dgamma":
    parameter_values = np.logspace(np.log10(0.01), np.log10(1.0), Nparam)
else:
    raise ValueError("SWEEP_PARAMETER must be 'nu' or 'dgamma'.")

Nmax = 3000
M = 10000
SEED = None

# Saturation convergence
x = 0.005
Ntail_tolerance = x * Nmax
max_iterations = max(1, Nmax // 100)


# ============================================================
# BASIS AND HAMILTONIAN
# ============================================================

L = np.array([1.0, 0.0], dtype=complex)
R = np.array([0.0, 1.0], dtype=complex)

sigz = np.array([[1, 0], [0, -1]], dtype=complex)
sigx = np.array([[0, 1], [1, 0]], dtype=complex)

P_l = np.outer(L, L.conj())
P_r = np.outer(R, R.conj())
id2 = np.eye(2, dtype=complex)

# Detector reset state is |B> = |L>.
P_B = P_l


def make_kraus_operators(epsilon, nu, gamma, dgamma, dt):
    """Return the two 2x2 system Kraus operators for outcomes B and T."""
    H_q = epsilon * sigz + nu * sigx
    H_d = gamma * sigx
    H_int = dgamma * np.kron(P_r, sigx)

    H_tot = (
        np.kron(H_q, id2)
        + np.kron(id2, H_d)
        + H_int
    )

    U = la.expm(-1j * H_tot * dt / hbar)

    # Joint basis: |L,B>, |L,T>, |R,B>, |R,T>.
    # The detector input is always |B>, so retain columns 0 and 2.
    V = U[:, [0, 2]]

    # Rows for detector outcome B are 0,2; for T are 1,3.
    K_B = V[[0, 2], :]
    K_T = V[[1, 3], :]

    # This should hold up to numerical precision.
    completeness = K_B.conj().T @ K_B + K_T.conj().T @ K_T
    if not np.allclose(completeness, id2, atol=1e-10):
        raise RuntimeError("Kraus operators fail the completeness check.")

    return K_B, K_T


# ============================================================
# DIRECT ML: VECTORIZED MONTE CARLO
# ============================================================

def simulate_all_lengths(Nmax, M, K_B, K_T, seed=None):
    """
    Monte Carlo estimate of direct-ML fidelity for N=1,...,Nmax.

    The true initial state is uniformly chosen from L/R. At each
    step the detector is reset to B, and both initial-state
    hypotheses are updated using the observed outcome.
    """
    rng = np.random.default_rng(seed)
    true_state = rng.integers(0, 2, size=M)

    psi = np.zeros((M, 2), dtype=complex)
    psi[np.arange(M), true_state] = 1.0

    # states[:, 0, :] and states[:, 1, :] are the normalized
    # conditional qubit states under initial-L and initial-R hypotheses.
    states = np.zeros((M, 2, 2), dtype=complex)
    states[:, 0, 0] = 1.0
    states[:, 1, 1] = 1.0

    logL = np.zeros(M)
    logR = np.zeros(M)
    results = np.empty(Nmax)

    for n in range(Nmax):
        # Generate the next actual outcome.
        cand_B = psi @ K_B.T
        cand_T = psi @ K_T.T

        pB = np.sum(np.abs(cand_B) ** 2, axis=1)
        pT = np.sum(np.abs(cand_T) ** 2, axis=1)

        # As in the original implementation, use the probabilities
        # directly and preserve the normalization check.
        total = pB + pT
        if np.max(np.abs(total - 1.0)) > 1e-8:
            raise RuntimeError(
                "Probability normalization failed during trajectory generation."
            )

        outcome = rng.random(M) < pT
        selected_true = np.where(outcome[:, None], cand_T, cand_B)
        prob_true = np.where(outcome, pT, pB)
        psi = selected_true / np.sqrt(
            np.maximum(prob_true, np.finfo(float).tiny)
        )[:, None]

        # Propagate both hypotheses directly with the Kraus operators.
        hyp_B = states @ K_B.T
        hyp_T = states @ K_T.T

        pB_h = np.sum(np.abs(hyp_B) ** 2, axis=2)
        pT_h = np.sum(np.abs(hyp_T) ** 2, axis=2)

        probs = np.where(outcome[:, None], pT_h, pB_h)

        with np.errstate(divide="ignore"):
            log_probs = np.log(probs)

        logL += log_probs[:, 0]
        logR += log_probs[:, 1]

        selected = np.where(
            outcome[:, None, None],
            hyp_T,
            hyp_B
        )

        valid = probs > 0.0
        denom = np.sqrt(np.where(valid, probs, 1.0))
        updated = selected / denom[:, :, None]

        # If a hypothesis gives exactly zero probability to the outcome,
        # keep its previous state; its log-likelihood is already -inf.
        states = np.where(valid[:, :, None], updated, states)

        guess = (logR >= logL).astype(int)
        results[n] = np.mean(guess == true_state)

    return results


# ============================================================
# EXACT AVERAGING-INFERENCE FIDELITY
# ============================================================

def averaging_fidelity_map(Nmax, K_B, K_T):
    """
    Exact fidelity for inference based only on the number of T outcomes.

    Density matrices are indexed by initial-state hypothesis and T count.
    Updates over all counts are vectorized; no Python loop over k.
    """
    current = np.zeros((2, Nmax + 1, 2, 2), dtype=complex)
    next_current = np.zeros_like(current)

    current[0, 0] = P_l
    current[1, 0] = P_r

    fidmap = np.empty(Nmax)
    K_B_conj = K_B.conj()
    K_T_conj = K_T.conj()

    for n in range(Nmax):
        # There are n measurements so far, hence counts 0,...,n.
        rho = current[:, :n + 1]

        # Vectorized K rho K† for both initial states and every count.
        # NumPy broadcasts the 2x2 Kraus matrices across all
        # (initial-state, count) density matrices.
        rho_B = K_B @ rho @ K_B.conj().T
        rho_T = K_T @ rho @ K_T.conj().T

        # Clear only the region written during this iteration.
        next_current[:, :n + 2] = 0.0
        next_current[:, :n + 1] = rho_B
        next_current[:, 1:n + 2] += rho_T

        current, next_current = next_current, current

        probs = np.trace(
            current[:, :n + 2], axis1=-2, axis2=-1
        ).real

        # Equal priors for L/R; choose the more likely initial state
        # for each possible number of T outcomes.
        fidmap[n] = 0.5 * np.maximum(probs[0], probs[1]).sum()

    return fidmap


# ============================================================
# SATURATION CUTOFF
# ============================================================

def estimate_saturation(results, M, tolerance, max_iterations):
    # The suffix ranges depend only on Results, not on delta, so
    # compute them once rather than once per convergence iteration.
    suffix_max = np.maximum.accumulate(results[::-1])[::-1]
    suffix_min = np.minimum.accumulate(results[::-1])[::-1]
    suffix_ranges = suffix_max - suffix_min

    # Match the original convergence loop's initial estimate exactly.
    Fsat = 0.5
    Ntail = len(results)
    converged = False

    for iteration in range(1, max_iterations + 1):
        delta = 2.0 * np.sqrt(max(Fsat * (1.0 - Fsat), 0.0) / M)

        eligible = np.flatnonzero(suffix_ranges <= delta)
        Ntail_new = (
            len(results) if eligible.size == 0
            else int(eligible[0] + 1)
        )
        Fsat_new = float(np.mean(results[Ntail_new - 1:]))

        cutoff_change = abs(Ntail_new - Ntail)
        Ntail = Ntail_new
        Fsat = Fsat_new

        if cutoff_change <= tolerance:
            converged = True
            break

    delta = 2.0 * np.sqrt(max(Fsat * (1.0 - Fsat), 0.0) / M)
    return Fsat, Ntail, iteration, delta, converged


# ============================================================
# PARAMETER SWEEP
# ============================================================

Fsat_values = np.zeros(Nparam)
Favgmax_values = np.zeros(Nparam)
Ntail_values = np.zeros(Nparam, dtype=int)
Navgmax_values = np.zeros(Nparam, dtype=int)
iterations_values = np.zeros(Nparam, dtype=int)
delta_values = np.zeros(Nparam)
converged_values = np.zeros(Nparam, dtype=bool)

for j, value in enumerate(parameter_values):
    if SWEEP_PARAMETER == "nu":
        nu_current = value
        dgamma_current = dgamma
    else:
        nu_current = nu
        dgamma_current = value

    print(f"{j + 1}/{Nparam}: {SWEEP_PARAMETER} = {value:.6g} μeV")

    K_B, K_T = make_kraus_operators(
        epsilon, nu_current, gamma, dgamma_current, dt
    )

    # Full-trajectory ML Monte Carlo
    Results = simulate_all_lengths(
        Nmax, M, K_B, K_T,
        seed=None if SEED is None else SEED + j
    )

    Fsat, Ntail, iteration, delta, converged = estimate_saturation(
        Results, M, Ntail_tolerance, max_iterations
    )

    # Exact averaging-inference curve
    fidmap = averaging_fidelity_map(Nmax, K_B, K_T)
    Favgind = int(np.argmax(fidmap))
    Favgmax = float(fidmap[Favgind])

    print(
        f"    Iterations = {iteration}, Ntail = {Ntail}, "
        f"Fsat = {Fsat:.8f}, delta = {delta:.3e}"
    )
    print(
        f"    Favg,max = {Favgmax:.8f}, Navg,max = {Favgind + 1}"
    )

    Fsat_values[j] = Fsat
    Favgmax_values[j] = Favgmax
    Ntail_values[j] = Ntail
    Navgmax_values[j] = Favgind + 1
    iterations_values[j] = iteration
    delta_values[j] = delta
    converged_values[j] = converged


# ============================================================
# SAVE RESULTS (same filenames as the supplied code)
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
# PLOTS
# ============================================================

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{braket}",
})

xlabel = (
    r"$\nu$ ($\mu\mathrm{eV}$)"
    if SWEEP_PARAMETER == "nu"
    else r"$\delta\gamma$ ($\mu\mathrm{eV}$)"
)

# Infidelity vs swept parameter
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(
    parameter_values, 1.0 - Fsat_values, "o-",
    label=r"Full-trajectory ML: $1-\mathcal{F}_{\mathrm{sat}}$"
)
ax.plot(
    parameter_values, 1.0 - Favgmax_values, "o-",
    label=r"Averaging: $1-\mathcal{F}_{\mathrm{avg,max}}$"
)
if np.all(parameter_values > 0):
    ax.set_xscale("log")
ax.set_xlabel(xlabel, fontsize=15)
ax.set_ylabel(r"$1-\mathcal{F}$", fontsize=15)
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(f"{prefix}_infidelity.png", dpi=200)

# Optimal lengths vs swept parameter
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(
    parameter_values, Ntail_values, "o-",
    label=r"$N_{\mathrm{tail}}$ (ML)"
)
ax.plot(
    parameter_values, Navgmax_values, "o-",
    label=r"$N_{\mathrm{avg,max}}$"
)
if np.all(parameter_values > 0):
    ax.set_xscale("log")
ax.set_xlabel(xlabel, fontsize=28)
ax.set_ylabel(r"$N$", fontsize=28)
ax.tick_params(axis="both", labelsize=22)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=14)
fig.tight_layout()
fig.savefig(f"{prefix}_optimal_lengths.png", dpi=200)

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
