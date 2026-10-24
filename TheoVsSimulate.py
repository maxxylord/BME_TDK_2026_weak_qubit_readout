
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS
# ============================================================

N = 5
M = 10_000

ps = np.linspace(0.5, 1.0, 100)
qs = np.linspace(0.0, 0.5, 100)

p_fixed = 0.7
q_fixed = 0.15

q_cut = np.linspace(0.0, 0.5, 50)
p_cut = np.linspace(0.5, 1.0, 50)

# Optional: reproducible Monte Carlo results
rng = np.random.default_rng(42)


# ============================================================
# MONTE CARLO GENERATOR
# ============================================================

def transgen(q, p, N, M, rng=rng):
    """Generate M trajectories and their initial hidden states."""

    start = rng.integers(0, 2, size=M)

    # Initial state, including the first transition
    hidden = np.empty((M, N), dtype=np.int8)
    hidden[:, 0] = start ^ (rng.random(M) < q)

    # Subsequent transitions
    flips = rng.random((M, N - 1)) < q

    for t in range(1, N):
        hidden[:, t] = hidden[:, t - 1] ^ flips[:, t - 1]

    # Generate measurement outcomes
    emission_p = np.where(hidden == 0, p, 1.0 - p)
    data = (rng.random((M, N)) < emission_p).astype(np.int8)

    return data, start


# ============================================================
# VECTORIZED HMM FORWARD EVALUATION
# ============================================================

def evaluate(M, N, p, q, rng=rng):
    """
    Estimate initial-state readout fidelity by Monte Carlo.

    Both initial-state hypotheses are evaluated simultaneously.
    Computation is vectorized over all M trajectories.
    """

    data, start = transgen(q, p, N, M, rng)

    # Log probabilities. -inf at zero probability is valid.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_q = np.log(q)
        log_1mq = np.log1p(-q)
        log_p = np.log(p)
        log_1mp = np.log1p(-p)

    # F[:, hidden_state, initial_state_hypothesis]
    # Incorporate the transition before the first observation.
    F = np.empty((M, 2, 2), dtype=float)

    F[:, 0, 0] = log_1mq  # hidden=0, hypothesized start=0
    F[:, 1, 0] = log_q
    F[:, 0, 1] = log_q
    F[:, 1, 1] = log_1mq

    # Emission probabilities for the first observation
    obs = data[:, 0]
    e0 = np.where(obs == 1, log_p, log_1mp)
    e1 = np.where(obs == 1, log_1mp, log_p)

    F[:, 0, :] += e0[:, None]
    F[:, 1, :] += e1[:, None]

    # Forward recursion for the remaining observations
    for t in range(1, N):
        obs = data[:, t]

        # Transition, using vectorized log-sum-exp
        new0 = np.logaddexp(
            F[:, 0, :] + log_1mq,
            F[:, 1, :] + log_q
        )

        new1 = np.logaddexp(
            F[:, 0, :] + log_q,
            F[:, 1, :] + log_1mq
        )

        # Emission
        e0 = np.where(obs == 1, log_p, log_1mp)
        e1 = np.where(obs == 1, log_1mp, log_p)

        F[:, 0, :] = new0 + e0[:, None]
        F[:, 1, :] = new1 + e1[:, None]

    # Marginalize over the final hidden state
    like0 = np.logaddexp(F[:, 0, 0], F[:, 1, 0])
    like1 = np.logaddexp(F[:, 0, 1], F[:, 1, 1])

    guesses = (like1 > like0).astype(np.int8)

    return np.mean(guesses == start)


# ============================================================
# EXACT THEORY
# ============================================================

def make_observations(N):
    """Enumerate every binary measurement record."""
    indices = np.arange(2**N, dtype=np.uint64)
    shifts = np.arange(N - 1, -1, -1, dtype=np.uint64)

    return ((indices[:, None] >> shifts) & 1).astype(np.int8)


def theory_fidelity(p, q, N, observations=None):
    """
    Exact optimal initial-state classification fidelity:
        F = 1/2 * sum_data max(P(data|start=0),
                              P(data|start=1)).
    """

    if observations is None:
        observations = make_observations(N)

    n_records = len(observations)

    def likelihood(start):
        # Deterministic initial state before the first transition
        prob0 = np.full(n_records, float(start == 0))
        prob1 = np.full(n_records, float(start == 1))

        for t in range(N):
            # Transition before each measurement
            new0 = prob0 * (1.0 - q) + prob1 * q
            new1 = prob0 * q + prob1 * (1.0 - q)

            # Measurement likelihood
            obs = observations[:, t]
            prob0 = new0 * np.where(obs == 1, p, 1.0 - p)
            prob1 = new1 * np.where(obs == 1, 1.0 - p, p)

        return prob0 + prob1

    likelihood0 = likelihood(0)
    likelihood1 = likelihood(1)

    return 0.5 * np.maximum(likelihood0, likelihood1).sum()


def theory_fidelity_grid(ps, qs, N):
    """Exact fidelity on the complete parameter grid."""

    observations = make_observations(N)
    out = np.empty((len(qs), len(ps)))

    for i, q in enumerate(qs):
        for j, p in enumerate(ps):
            out[i, j] = theory_fidelity(
                p, q, N, observations
            )

    return out


# ============================================================
# SIMULATION AND THEORY ON THE PARAMETER GRID
# ============================================================

ml_sim = np.empty((len(qs), len(ps)))

for i, q in enumerate(qs):
    for j, p in enumerate(ps):
        ml_sim[i, j] = evaluate(M, N, p, q)

ml_theo = theory_fidelity_grid(ps, qs, N)


# ============================================================
# 1D CUTS
# ============================================================

observations = make_observations(N)

# Fixed p, varying q
theo_p_fixed = np.array([
    theory_fidelity(p_fixed, q, N, observations)
    for q in q_cut
])

sim_p_fixed = np.array([
    evaluate(M, N, p_fixed, q)
    for q in q_cut
])

# Fixed q, varying p
theo_q_fixed = np.array([
    theory_fidelity(p, q_fixed, N, observations)
    for p in p_cut
])

sim_q_fixed = np.array([
    evaluate(M, N, p, q_fixed)
    for p in p_cut
])


# ============================================================
# PLOTTING
# ============================================================

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{braket}",
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax in axes:
    ax.tick_params(axis="both", which="major", labelsize=26)
    ax.set_ylabel(r"$\mathcal{F}$", fontsize=32)
    ax.legend()
    ax.grid(alpha=0.3)

# Fixed p = 0.7: F(q)
axes[0].plot(q_cut, theo_p_fixed, color="blue", label="Exact")
axes[0].plot(
    q_cut, sim_p_fixed,
    color="red", linestyle="--", label="Monte Carlo"
)
axes[0].set_title(r"$p=0.7$")
axes[0].set_xlabel(r"$q$", fontsize=26)
axes[0].legend(fontsize=20)

# Fixed q = 0.15: F(p)
axes[1].plot(p_cut, theo_q_fixed, color="blue", label="Exact")
axes[1].plot(
    p_cut, sim_q_fixed,
    color="red", linestyle="--", label="Monte Carlo"
)
axes[1].set_title(r"$q=0.15$")
axes[1].set_xlabel(
    r"$p$", fontsize=26
)
axes[1].legend(fontsize=20)

plt.tight_layout()
plt.show()

# ============================================================
# DENSITY PLOTS: F(p, q)
# ============================================================

P, Q = np.meshgrid(ps, qs)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{braket}",
})

# Exact theoretical fidelity
im0 = axes[0].pcolormesh(
    P, Q, ml_theo,
    shading="auto",
    cmap="viridis",
    vmin=0.5, vmax=1.0
)
axes[0].set_title("Exact theory", fontsize=24)
axes[0].set_xlabel(r"$p$", fontsize=26)
axes[0].set_ylabel(r"$q$", fontsize=26)
axes[0].tick_params(axis="both", labelsize=18)

cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.set_label(r"$\mathcal{F}(p,q)$", fontsize=22)
cbar0.ax.tick_params(labelsize=16)

# Monte Carlo fidelity
im1 = axes[1].pcolormesh(
    P, Q, ml_sim,
    shading="auto",
    cmap="viridis",
    vmin=0.5, vmax=1.0
)
axes[1].set_title("Monte Carlo", fontsize=24)
axes[1].set_xlabel(r"$p$", fontsize=26)
axes[1].set_ylabel(r"$q$", fontsize=26)
axes[1].tick_params(axis="both", labelsize=18)

cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.set_label(r"$\mathcal{F}(p,q)$", fontsize=22)
cbar1.ax.tick_params(labelsize=16)

plt.show()
