
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS
# ============================================================

N = 5

ps = np.linspace(0.5, 1.0, 100)
qs = np.linspace(0.0, 0.5, 100)

p_fixed = 0.7
q_fixed = 0.15

q_cut = np.linspace(0.0, 0.5, 50)
p_cut = np.linspace(0.5, 1.0, 50)


# ============================================================
# ENUMERATE ALL BINARY MEASUREMENT RECORDS
# ============================================================

def make_observations(N):
    """Return all binary measurement records of length N."""

    indices = np.arange(2**N, dtype=np.uint64)
    shifts = np.arange(N - 1, -1, -1, dtype=np.uint64)

    return ((indices[:, None] >> shifts) & 1).astype(np.int8)


# ============================================================
# EXACT FULL-ML INFERENCE FIDELITY
# ============================================================

def theory_fidelity(p, q, N, observations=None):
    """
    Exact full-ML initial-state classification fidelity.

    F_ML = 1/2 * sum_x max(P(x|0), P(x|1)).

    The hidden state undergoes a transition before every
    measurement, including the first one.
    """

    if observations is None:
        observations = make_observations(N)

    n_records = len(observations)

    def likelihood(start):
        # Probabilities for the two hidden states
        prob0 = np.full(n_records, float(start == 0))
        prob1 = np.full(n_records, float(start == 1))

        for t in range(N):
            # Transition before measurement
            new0 = prob0 * (1.0 - q) + prob1 * q
            new1 = prob0 * q + prob1 * (1.0 - q)

            # Measurement likelihood
            obs = observations[:, t]

            prob0 = new0 * np.where(
                obs == 1, p, 1.0 - p
            )

            prob1 = new1 * np.where(
                obs == 1, 1.0 - p, p
            )

        return prob0 + prob1

    likelihood0 = likelihood(0)
    likelihood1 = likelihood(1)

    return 0.5 * np.maximum(
        likelihood0, likelihood1
    ).sum()


# ============================================================
# EXACT AVERAGE-TIME-SERIES INFERENCE FIDELITY
# ============================================================

def average_fidelity(p, q, N):
    """
    Exact initial-state classification fidelity using only
    the number of measurement outcomes equal to 1.

    Returns the fidelity after exactly N measurements.
    """

    # Transition-before-measurement matrices.
    # Rows: previous hidden state.
    # Columns: next hidden state, with emission included.

    A = np.array([
        [(1.0 - p) * (1.0 - q), q * p],
        [q * (1.0 - p), (1.0 - q) * p]
    ])

    B = np.array([
        [p * (1.0 - q), q * (1.0 - p)],
        [q * p, (1.0 - q) * (1.0 - p)]
    ])

    # Row i stores probabilities for i transmissions,
    # resolved by the hidden state after the measurement.

    left = np.zeros((N + 1, 2))
    right = np.zeros((N + 1, 2))

    left[0] = [1.0, 0.0]
    right[0] = [0.0, 1.0]

    for k in range(N):
        next_left = np.zeros_like(left)
        next_right = np.zeros_like(right)

        for i in range(k + 2):

            if i == 0:
                next_left[i] = left[i] @ A
                next_right[i] = right[i] @ A

            elif i == k + 1:
                next_left[i] = left[i - 1] @ B
                next_right[i] = right[i - 1] @ B

            else:
                next_left[i] = (
                    left[i] @ A
                    + left[i - 1] @ B
                )

                next_right[i] = (
                    right[i] @ A
                    + right[i - 1] @ B
                )

        left = next_left
        right = next_right

    # Marginalize over the final hidden state.
    left_prob = left.sum(axis=1)
    right_prob = right.sum(axis=1)

    # Maximum-likelihood classification using the count only.
    return 0.5 * np.maximum(
        left_prob, right_prob
    ).sum()


# ============================================================
# EXACT FIDELITIES ON THE PARAMETER GRID
# ============================================================

observations = make_observations(N)

ml_theo = np.empty((len(qs), len(ps)))
avg_theo = np.empty_like(ml_theo)

for i, q in enumerate(qs):
    for j, p in enumerate(ps):

        ml_theo[i, j] = theory_fidelity(
            p, q, N, observations
        )

        avg_theo[i, j] = average_fidelity(
            p, q, N
        )


# ============================================================
# ONE-DIMENSIONAL PARAMETER CUTS
# ============================================================

# Fixed p, varying q
ml_p_fixed = np.array([
    theory_fidelity(p_fixed, q, N, observations)
    for q in q_cut
])

avg_p_fixed = np.array([
    average_fidelity(p_fixed, q, N)
    for q in q_cut
])

# Fixed q, varying p
ml_q_fixed = np.array([
    theory_fidelity(p, q_fixed, N, observations)
    for p in p_cut
])

avg_q_fixed = np.array([
    average_fidelity(p, q_fixed, N)
    for p in p_cut
])


# ============================================================
# PLOTTING SETTINGS
# ============================================================

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": (
        r"\usepackage{amsmath}"
        r"\usepackage{braket}"
    ),
})


# ============================================================
# 1. PARAMETER CUTS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax in axes:
    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.set_ylabel(r"$\mathcal{F}$", fontsize=30)
    ax.grid(alpha=0.3)

# Fixed p = 0.7
axes[0].plot(
    q_cut, ml_p_fixed,
    color="blue", label="Full ML"
)

axes[0].plot(
    q_cut, avg_p_fixed,
    color="red", linestyle="--", label="Averaging"
)

axes[0].set_title(r"$p=0.7$", fontsize=32)
axes[0].set_xlabel(r"$q$", fontsize=32)
axes[0].legend(fontsize=22)

# Fixed q = 0.15
axes[1].plot(
    p_cut, ml_q_fixed,
    color="blue", label="Full ML"
)

axes[1].plot(
    p_cut, avg_q_fixed,
    color="red", linestyle="--", label="Averaging"
)

axes[1].set_title(r"$q=0.15$", fontsize=32)
axes[1].set_xlabel(r"$p$", fontsize=32)
axes[1].legend(fontsize=22)

plt.tight_layout()
plt.show()


# ============================================================
# 2. DENSITY PLOTS: F(p, q)
# ============================================================

P, Q = np.meshgrid(ps, qs)

fig, axes = plt.subplots(
    1, 2, figsize=(16, 6), constrained_layout=True
)

# Full ML fidelity
im0 = axes[0].pcolormesh(
    P, Q, ml_theo,
    shading="auto",
    cmap="viridis",
    vmin=0.5,
    vmax=1.0
)

axes[0].set_title("Full ML inference", fontsize=32)
axes[0].set_xlabel(r"$p$", fontsize=32)
axes[0].set_ylabel(r"$q$", fontsize=32)
axes[0].tick_params(axis="both", labelsize=22)

cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.set_label(
    r"$\mathcal{F}_{\mathrm{ML}}(p,q)$", fontsize=22
)
cbar0.ax.tick_params(labelsize=16)

# Average-time-series fidelity after exactly N measurements
im1 = axes[1].pcolormesh(
    P, Q, avg_theo,
    shading="auto",
    cmap="viridis",
    vmin=0.5,
    vmax=1.0
)

axes[1].set_title(
    rf"Averaging inference", fontsize=32
)
axes[1].set_xlabel(r"$p$", fontsize=32)
axes[1].set_ylabel(r"$q$", fontsize=32)
axes[1].tick_params(axis="both", labelsize=22)

cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.set_label(
    r"$\mathcal{F}_{\mathrm{avg}}(p,q)$", fontsize=22
)
cbar1.ax.tick_params(labelsize=16)

plt.show()


# ============================================================
# 3. FIDELITY DIFFERENCE
# ============================================================

difference = ml_theo - avg_theo

fig, ax = plt.subplots(figsize=(8, 6))

im = ax.pcolormesh(
    P, Q, difference,
    shading="auto",
    cmap="magma",
    vmin=0.0,
    vmax=max(1e-12, difference.max())
)

ax.set_title("Full ML advantage", fontsize=32)
ax.set_xlabel(r"$p$", fontsize=32)
ax.set_ylabel(r"$q$", fontsize=32)
ax.tick_params(axis="both", labelsize=22)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(
    r"$\mathcal{F}_{\mathrm{ML}}-\mathcal{F}_{\mathrm{avg}}$",
    fontsize=20
)

plt.tight_layout()
plt.show()


# ============================================================
# 4. BASIC CONSISTENCY CHECK
# ============================================================

print(f"N = {N}")
print(f"Maximum ML advantage: {difference.max():.12g}")
print(f"Minimum ML advantage: {difference.min():.12g}")

if difference.min() < -1e-10:
    print(
        "WARNING: Averaging fidelity exceeds full ML fidelity. "
        "Check the probability-model conventions."
    )
else:
    print(
        "Consistency check passed: full ML fidelity is "
        "at least as large as averaging fidelity "
        "up to numerical precision."
    )
