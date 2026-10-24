
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS
# ============================================================

p = 0.55

qmin = 1e-5
qmax = 5e-3
Nq = 30

q_values = np.logspace(np.log10(qmin), np.log10(qmax), Nq)

Nmax = 1000
M = 10000

# Saturation convergence parameters
x = 0.005                  # 0.5% of Nmax
Ntail_tolerance = x * Nmax
max_iterations = max(1, Nmax // 100)


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
# DIRECT ML MONTE CARLO
#
# Returns fidelity after every N = 1,...,Nmax
# ============================================================

def evaluate(M, N, p, q):

    data, start = transgen(q, p, N, M)

    log_q = np.log(q + 1e-300)
    log_1mq = np.log(1 - q + 1e-300)

    log_p = np.log(p + 1e-300)
    log_1mp = np.log(1 - p + 1e-300)

    F = np.full((M, 2, 2), -np.inf)

    F[:, 0, 0] = 0.0
    F[:, 1, 1] = 0.0

    Results = np.zeros(N)

    for t in range(N):

        x_obs = data[:, t]

        eL = np.where(x_obs == 1, log_p, log_1mp)
        eR = np.where(x_obs == 1, log_1mp, log_p)

        newF = np.empty_like(F)

        # Final hidden state = L

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

        # Final hidden state = R

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

        like0 = logsumexp(
            F[:, 0, 0],
            F[:, 1, 0]
        )

        like1 = logsumexp(
            F[:, 0, 1],
            F[:, 1, 1]
        )

        guesses = (like1 > like0).astype(int)

        Results[t] = np.mean(guesses == start)

    return Results


# ============================================================
# EXACT AVERAGING-INFERENCE FIDELITY
# ============================================================

def fidelity(p, q, N):

    A = np.array([
        [(1-p)*(1-q), (1-p)*q],
        [q*p,         (1-q)*p]
    ])

    B = np.array([
        [p*(1-q),     q*p],
        [q*(1-p),     (1-q)*(1-p)]
    ])

    left = np.zeros((N + 1, 2))
    right = np.zeros((N + 1, 2))

    left[0] = [1.0, 0.0]
    right[0] = [0.0, 1.0]

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

        left_prob = np.sum(left, axis=1)
        right_prob = np.sum(right, axis=1)

        fidelities[k] = 0.5 * np.sum(
            np.maximum(left_prob, right_prob)
        )

    return fidelities


# ============================================================
# ARRAYS FOR RESULTS
# ============================================================

Fsat_values = np.zeros(Nq)
Favgmax_values = np.zeros(Nq)

Ntail_values = np.zeros(Nq, dtype=int)
Navgmax_values = np.zeros(Nq, dtype=int)

# Optional diagnostics
iterations_values = np.zeros(Nq, dtype=int)
delta_values = np.zeros(Nq)


# ============================================================
# Q SWEEP
# ============================================================

for j, q in enumerate(q_values):

    print(f"{j+1}/{Nq}: q = {q:.5e}")

    # --------------------------------------------------------
    # Direct ML
    # --------------------------------------------------------

    Results = evaluate(M, Nmax, p, q)

    # --------------------------------------------------------
    # Iterative saturation estimate
    #
    # Start with the fidelity-independent upper bound:
    # sigma <= 1 / (2 sqrt(M))
    #
    # At each iteration:
    #   1. Set delta = 2 sigma using the current Fsat.
    #   2. Find the tail satisfying the max-min criterion.
    #   3. Update Fsat from the mean of that tail.
    #   4. Stop when Ntail converges within 0.5% of Nmax.
    # --------------------------------------------------------

    Fsat = 0.5
    Ntail = Nmax
    Ntail_prev = None
    converged = False

    for iteration in range(1, max_iterations + 1):

        delta = 2 * np.sqrt(Fsat * (1 - Fsat) / M)

        Fmax = Results[-1]
        Fmin = Results[-1]
        Ntail_new = Nmax

        for i in range(Nmax - 1, 0, -1):

            Fmax = max(Fmax, Results[i])
            Fmin = min(Fmin, Results[i])

            if Fmax - Fmin > delta:
                Ntail_new = i + 1
                break

        # Update the cutoff and estimate from the new tail
        Ntail_prev = Ntail
        Ntail = Ntail_new

        Fsat = np.mean(Results[Ntail - 1:])

        # Test convergence of consecutive cutoff estimates
        if (
            Ntail_prev is not None
            and abs(Ntail - Ntail_prev) <= Ntail_tolerance
        ):
            converged = True
            break

    # Final uncertainty scale, based on final Fsat
    delta = 2 * np.sqrt(Fsat * (1 - Fsat) / M)

    if not converged:
        print(
            f"    WARNING: did not converge within "
            f"{max_iterations} iterations."
        )

    print(
        f"    Iterations = {iteration}, "
        f"Ntail = {Ntail}, "
        f"Fsat = {Fsat:.8f}, "
        f"delta = {delta:.3e}"
    )

    # --------------------------------------------------------
    # Exact averaging-inference result
    # --------------------------------------------------------

    fidmap = fidelity(p, q, Nmax)

    Favgind = np.argmax(fidmap)
    Favgmax = fidmap[Favgind]

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    Fsat_values[j] = Fsat
    Favgmax_values[j] = Favgmax

    Ntail_values[j] = Ntail
    Navgmax_values[j] = Favgind + 1

    iterations_values[j] = iteration
    delta_values[j] = delta


# ============================================================
# SAVE
# ============================================================

'''
np.save("q_values_p055.npy", q_values)
np.save("Fsat_p055.npy", Fsat_values)
np.save("Favgmax_p055.npy", Favgmax_values)
np.save("Ntail_p055.npy", Ntail_values)
np.save("Navgmax_p055.npy", Navgmax_values)
np.save("iterations_p055.npy", iterations_values)
np.save("delta_p055.npy", delta_values)
'''


# ============================================================
# FIDELITY VS q
# ============================================================
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{braket}",
})
plt.figure()
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.semilogx(
    q_values,
    1 - Fsat_values,
    'o-',
    label='Direct ML: $1-\mathcal{F}^{d}_{\mathrm{sat}}$'
)

plt.semilogx(
    q_values,
    1 - Favgmax_values,
    'o-',
    label='Averaging: $1-\mathcal{F}^{\mathrm{avg}}_{\mathrm{max}}$'
)

plt.xlabel('$q$',fontsize=32)
plt.ylabel('$1-\mathcal{F}$',fontsize=32)
plt.grid()
plt.legend(fontsize=26)

plt.show()


# ============================================================
# OPTIMAL N VS q
# ============================================================

plt.figure()

plt.semilogx(
    q_values,
    Ntail_values,
    'o-',
    label='$N_{tail}$'
)

plt.semilogx(
    q_values,
    Navgmax_values,
    'o-',
    label='$N_{max,avg}$'
)

plt.xlabel('$q$')
plt.ylabel('$N$')
plt.grid()
plt.legend()

plt.show()
