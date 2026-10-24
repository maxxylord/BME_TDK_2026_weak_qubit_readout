
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

SWEEP_PARAMETER = "dgamma"  # "nu" or "dgamma"

prefix = f"Quantum_{SWEEP_PARAMETER}"

# ============================================================
# LOAD SAVED DATA
# ============================================================

parameter_values = np.load(f"{prefix}_parameter_values.npy")
Fsat = np.load(f"{prefix}_Fsat.npy")
Favgmax = np.load(f"{prefix}_Favgmax.npy")
Ntail = np.load(f"{prefix}_Ntail.npy")
Navgmax = np.load(f"{prefix}_Navgmax.npy")

# Convert fidelities to infidelities.
ml_infidelity = 1.0 - Fsat
avg_infidelity = 1.0 - Favgmax

zero = parameter_values == 0
positive = parameter_values > 0

# ============================================================
# PLOT 1: INFIDELITY
# ============================================================
if SWEEP_PARAMETER == "nu":
    SWEEP_PARAMETER = "t"
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    parameter_values[positive],
    ml_infidelity[positive],
    "o-",
    label="Full-trajectory ML"
)

ax.plot(
    parameter_values[positive],
    avg_infidelity[positive],
    "s-",
    label="Averaging"
)

ax.set_xscale("log")
#ax.set_xlabel(rf"${SWEEP_PARAMETER}$ ($\mu$eV)")
ax.set_xlabel(rf"$\delta \gamma$ ($\mu$eV)")
ax.set_yscale("log")
ax.set_ylabel("$1-\mathcal{F}$")
#ax.set_title("Readout infidelity vs parameter")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

if np.any(zero):
    i = np.flatnonzero(zero)[0]
    ax.text(
        0.03, 0.97,
        rf"$\mathbf{{{SWEEP_PARAMETER}=0}}$" + "\n"
        + rf"ML: {ml_infidelity[i]:.4g}" + "\n"
        + rf"Averaging: {avg_infidelity[i]:.4g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

fig.tight_layout()
fig.savefig(f"{prefix}_infidelity.png", dpi=300)

# ============================================================
# PLOT 2: OPTIMAL SEQUENCE LENGTH
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    parameter_values[positive],
    Ntail[positive],
    "o-",
    label="Full-trajectory ML"
)

ax.plot(
    parameter_values[positive],
    Navgmax[positive],
    "s-",
    label="Averaging"
)

ax.set_xscale("linear")
ax.set_xlabel(rf"${SWEEP_PARAMETER}$ ($\mu$eV)")
ax.set_ylabel("Optimal sequence length $N$")
ax.set_title("Optimal sequence length vs parameter")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

if np.any(zero):
    i = np.flatnonzero(zero)[0]
    ax.text(
        0.03, 0.97,
        rf"$\mathbf{{{SWEEP_PARAMETER}=0}}$" + "\n"
        + rf"ML: {Ntail[i]}" + "\n"
        + rf"Averaging: {Navgmax[i]}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

fig.tight_layout()
fig.savefig(f"{prefix}_optimal_lengths.png", dpi=300)

plt.show()
