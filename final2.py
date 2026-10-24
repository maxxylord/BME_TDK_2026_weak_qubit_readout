import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# GENERATOR (unchanged, correct physical model)
# ============================================================  
def transgen(q, p, N, M):

    start = np.random.randint(0, 2, size=M)

    hidden = np.zeros((M, N), dtype=int)
    hidden[:, 0] = start ^ (np.random.rand(M) < q)

    flips = np.random.rand(M, N) < q

    for t in range(1, N):
        hidden[:, t] = hidden[:, t - 1] ^ flips[:, t]

    noise = np.random.rand(M, N)

    data = np.zeros((M, N), dtype=int)

    data[hidden == 0] = (noise[hidden == 0] < p)
    data[hidden == 1] = (noise[hidden == 1] < (1 - p))

    return data, start


# ============================================================
# STABLE LOG-SUM-EXP
# ============================================================
def logsumexp(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


# ============================================================
# FIXED: TRUE HMM FORWARD EVALUATION (REPLACES OLD evaluate)
# ============================================================
# ============================================================
# PARALLEL HMM FORWARD EVALUATION
# ============================================================
def evaluate(M, N, p, q):

    data, start = transgen(q, p, N, M)

    # transition probabilities
    log_q = np.log(q + 1e-300)
    log_1mq = np.log(1 - q + 1e-300)

    # emission probabilities
    log_p = np.log(p + 1e-300)
    log_1mp = np.log(1 - p + 1e-300)

    guesses = np.zeros(M, dtype=int)

    for i in range(M):

        obs = data[i]

        # ----------------------------------------------------
        # F contains both initial-state hypotheses:
        #
        #             start=0     start=1
        # hidden=L       1            0
        # hidden=R       0            1
        #
        # We work in log space.
        # ----------------------------------------------------
        F = np.array([
            [0.0,    -np.inf],
            [-np.inf, 0.0]
        ])

        for x in obs:

            # emission probabilities
            if x == 1:
                eL, eR = log_p, log_1mp
            else:
                eL, eR = log_1mp, log_p

            # ------------------------------------------------
            # Transition + emission
            #
            # Each column is one initial-state hypothesis.
            # ------------------------------------------------

            newF = np.empty((2, 2))

            # final hidden state = L
            newF[0, 0] = (
                logsumexp(
                    F[0, 0] + log_1mq,
                    F[1, 0] + log_q
                ) + eL
            )

            newF[0, 1] = (
                logsumexp(
                    F[0, 1] + log_1mq,
                    F[1, 1] + log_q
                ) + eL
            )

            # final hidden state = R
            newF[1, 0] = (
                logsumexp(
                    F[0, 0] + log_q,
                    F[1, 0] + log_1mq
                ) + eR
            )

            newF[1, 1] = (
                logsumexp(
                    F[0, 1] + log_q,
                    F[1, 1] + log_1mq
                ) + eR
            )

            F = newF

        # ----------------------------------------------------
        # Marginalize over the final hidden state
        # ----------------------------------------------------

        like0 = logsumexp(F[0, 0], F[1, 0])
        like1 = logsumexp(F[0, 1], F[1, 1])

        # Maximum-likelihood estimate of the initial state
        guesses[i] = int(like1 > like0)

    return np.mean(guesses == start)



# ============================================================
# GRID
# ============================================================
def evaluate_grid(M, N, ps, qs):

    out = np.zeros((len(qs), len(ps)))

    for i, q in enumerate(qs):
        for j, p in enumerate(ps):
            out[i, j] = evaluate(M, N, p, q)

    return out


# ============================================================
# THEORY (UNCHANGED)
# ============================================================
def theory_fidelity_grid(ps, qs, N):

    out = np.zeros((len(qs), len(ps)))

    all_obs = np.array(
        [[int(x) for x in format(i, f'0{N}b')] for i in range(2**N)]
    )

    for i, q in enumerate(qs):
        for j, p in enumerate(ps):

            total = 0.0

            for obs in all_obs:

                def lik(start):
                    probL = 1.0 if start == 0 else 0.0
                    probR = 1.0 - probL

                    for o in obs:
                        newL = probL * (1 - q) + probR * q
                        newR = probL * q + probR * (1 - q)

                        if o == 1:
                            newL *= p
                            newR *= (1 - p)
                        else:
                            newL *= (1 - p)
                            newR *= p

                        probL, probR = newL, newR

                    return probL + probR

                pL = lik(0)
                pR = lik(1)

                total += 0.5 * max(pL, pR)

            out[i, j] = total

    return out

# ============================================================
# 4. PARAMETERS
# ============================================================
N = 5
M = 10000

ps = np.linspace(0.5, 1.0, 25)
qs = np.linspace(0.0, 0.5, 25)


# ============================================================
# 5. SIMULATION
# ============================================================
ml_sim = np.zeros((len(qs), len(ps)))

for i, q in enumerate(qs):
    for j, p in enumerate(ps):
        ml_sim[i, j] = evaluate(M, N, p, q)


# ============================================================
# 6. THEORY
# ============================================================
ml_theo = theory_fidelity_grid(ps, qs, N)


# ============================================================
# 7. PLOTTING
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))


c1 = axes[0].contourf(ps, qs, ml_theo, levels=50, cmap="viridis")
fig.colorbar(c1, ax=axes[0])
axes[0].set_title("Theoretical ML Fidelity")
axes[0].set_xlabel("p = P(transmit | L)")
axes[0].set_ylabel("q (tunneling)")


c2 = axes[1].contourf(ps, qs, ml_sim, levels=50, cmap="viridis")
fig.colorbar(c2, ax=axes[1])
axes[1].set_title("Simulated ML Fidelity")
axes[1].set_xlabel("p = P(transmit | L)")
axes[1].set_ylabel("q (tunneling)")


plt.suptitle("QMQ Charge Qubit Readout Fidelity (N=5)")
plt.tight_layout()
plt.show()