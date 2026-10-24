import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# GENERATOR
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

    return m + np.log(
        np.exp(a - m) + np.exp(b - m)
    )


# ============================================================
# HMM EVALUATION
#
# Results[n-1] = ML fidelity after n measurements.
# ============================================================
def evaluate(M, N, p, q):

    data, start = transgen(q, p, N, M)

    # Transition probabilities
    log_q = np.log(q + 1e-300)
    log_1mq = np.log(1 - q + 1e-300)

    # Emission probabilities
    log_p = np.log(p + 1e-300)
    log_1mp = np.log(1 - p + 1e-300)

    # --------------------------------------------------------
    # F[:, hidden_state, initial_state]
    #
    # F[:,0,0] = P(observations, hidden=L | initial=L)
    # F[:,1,0] = P(observations, hidden=R | initial=L)
    #
    # F[:,0,1] = P(observations, hidden=L | initial=R)
    # F[:,1,1] = P(observations, hidden=R | initial=R)
    # --------------------------------------------------------

    F = np.full((M, 2, 2), -np.inf)

    # Initial-state hypotheses
    F[:, 0, 0] = 0.0
    F[:, 1, 1] = 0.0

    Results = np.zeros(N)

    for t in range(N):

        x = data[:, t]

        # ----------------------------------------------------
        # Emission probabilities
        # ----------------------------------------------------

        eL = np.where(x == 1, log_p, log_1mp)
        eR = np.where(x == 1, log_1mp, log_p)

        newF = np.empty_like(F)

        # ----------------------------------------------------
        # Final hidden state = L
        # ----------------------------------------------------

        newF[:, 0, 0] = (
            logsumexp(
                F[:, 0, 0] + log_1mq,
                F[:, 1, 0] + log_q
            )
            + eL
        )

        newF[:, 0, 1] = (
            logsumexp(
                F[:, 0, 1] + log_1mq,
                F[:, 1, 1] + log_q
            )
            + eL
        )

        # ----------------------------------------------------
        # Final hidden state = R
        # ----------------------------------------------------

        newF[:, 1, 0] = (
            logsumexp(
                F[:, 0, 0] + log_q,
                F[:, 1, 0] + log_1mq
            )
            + eR
        )

        newF[:, 1, 1] = (
            logsumexp(
                F[:, 0, 1] + log_q,
                F[:, 1, 1] + log_1mq
            )
            + eR
        )

        F = newF

        # ----------------------------------------------------
        # Marginalize over final hidden state
        # ----------------------------------------------------

        like0 = logsumexp(
            F[:, 0, 0],
            F[:, 1, 0]
        )

        like1 = logsumexp(
            F[:, 0, 1],
            F[:, 1, 1]
        )

        # ----------------------------------------------------
        # ML estimate of initial state
        # ----------------------------------------------------

        guesses = (like1 > like0).astype(int)

        # Fidelity after this measurement
        Results[t] = np.mean(guesses == start)

    return Results


# ============================================================
# PARAMETERS
# ============================================================

p= 0.55
q= 10**(-5)

Nmax= 1000
M= 10000

# One evaluation gives F(N) for every N = 1,...,Nmax
Results = evaluate(M, Nmax, p, q)

Nlist = np.arange(1, Nmax + 1)


# ============================================================
# TRANSITION MATRICES
# ============================================================

A = np.array([
    [(1-p)*(1-q), (1-p)*q],
    [q*p,         (1-q)*p]
])

B = np.array([
    [p*(1-q),     q*p],
    [q*(1-p),     (1-q)*(1-p)]
])


# ============================================================
# CALCULATE FIDELITY AFTER EVERY ITERATION
# ============================================================

def fidelity(N):

    # --------------------------------------------------------
    # current[i, state]
    #
    # Probability of having i transmissions after the current
    # number of measurements, with the corresponding hidden
    # state.
    # --------------------------------------------------------

    left = np.zeros((N + 1, 2))
    right = np.zeros((N + 1, 2))

    left[0] = np.array([1.0, 0.0])
    right[0] = np.array([0.0, 1.0])

    fidelities = np.zeros(N)

    for k in range(N):

        next_left = np.zeros((N + 1, 2))
        next_right = np.zeros((N + 1, 2))

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

        # ----------------------------------------------------
        # Marginal probability of each observation count
        # ----------------------------------------------------

        left_prob = np.sum(left, axis=1)
        right_prob = np.sum(right, axis=1)

        # ----------------------------------------------------
        # Maximum-likelihood classification:
        #
        # For a given number of transmissions i, choose the
        # initial state with the larger likelihood.
        # ----------------------------------------------------

        fidelities[k] = 0.5 * np.sum(
            np.maximum(left_prob, right_prob)
        )

    return fidelities


# ============================================================
# CALCULATE ONCE UP TO Nmax
# ============================================================

fidmap = fidelity(Nmax)
Favgmax=np.max(fidmap)
Favgind=np.argmax(fidmap)

# ============================================================
# PLOT
# ============================================================
# ============================================================
# SATURATION ESTIMATION: K ITERATIONS
# ============================================================

# ============================================================
# SATURATION ESTIMATION: ITERATE UNTIL Ntail CONVERGES
# ============================================================

x = 0.005  # 0.5% of Nmax
Fsat = 0.5
Ntail = Nmax
Ntail_prev = None
iteration = 0

while Ntail_prev is None or abs(Ntail - Ntail_prev) > x * Nmax:

    Ntail_prev = Ntail
    iteration += 1

    # Updated standard deviation estimate
    delta = 2 * np.sqrt(Fsat * (1 - Fsat) / M)

    Fmax = Results[-1]
    Fmin = Results[-1]
    Ntail = Nmax

    for i in range(Nmax - 1, 1, -1):

        if Results[i] > Fmax:
            Fmax = Results[i]
        elif Results[i] < Fmin:
            Fmin = Results[i]

        if Fmax - Fmin > delta:
            Ntail = i
            break

    # Refine the saturation estimate
    Fsat = np.mean(Results[Ntail:])

    print(
        f"Iteration {iteration}: "
        f"Ntail = {Ntail}, "
        f"Fsat = {Fsat:.8f}, "
        f"delta = {delta:.3e}"
    )

# Recalculate delta for the final estimate
delta = 2 * np.sqrt(Fsat * (1 - Fsat) / M)

print(f"\nConverged: Ntail = {Ntail}, Fsat = {Fsat:.8f}")
#print(Ntail)
plt.figure(figsize=(12, 8))
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{braket}",
})
plt.yscale('log')
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.plot(Nlist, 1-Results,label='Direct ML')
plt.plot(Nlist, 1-fidmap,color='red', label='Averaging ML')
plt.scatter(Nlist[Favgind], 1-Favgmax, color='purple', label='$1-\mathcal{F}_{\mathrm{max}}^{\mathrm{avg}}=$'+f'{Favgmax:.3E}', s=100)
plt.grid()
plt.xlabel("$N$",fontsize=32)
plt.ylabel("$1-\mathcal{F}$",fontsize=32)
#plt.title(f"p={p} q={q:.2E} Fsat={Fsat:.3E} margin={delta:.2E}")
plt.hlines(1-Fsat, 0, Nmax, color='orange', label='$1-\mathcal{F}^{\mathrm{d}}_{\mathrm{sat}}=$'+f'{Fsat:.3E}',linestyle='dashed')
plt.hlines(1-Fsat+delta, 0, Nmax, color='green', label='$1-\mathcal{F}^{\mathrm{d}}_{\mathrm{sat}} + \sigma^{(1)}$',linestyle='dashed')
plt.hlines(max(1-Fsat-delta,0), 0, Nmax, color='green', label='$1-\mathcal{F}^{\mathrm{d}}_{\mathrm{sat}} - \sigma^{(1)}$',linestyle='dashed')
plt.legend(fontsize=16)
print(Nlist[Favgind], 1-Favgmax)
plt.show()
