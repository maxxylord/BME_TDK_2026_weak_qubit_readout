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
def evaluate_max(M, N, p, q):

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
# 4. PARAMETERS
# ============================================================
p = 0.55
q = 10**(-3)
Nmax = 500
M = 10000
Samples=50
Nlist=np.linspace(10,Nmax,Samples).astype(int)
print(Nlist)
Results=np.zeros(Samples)
for i in range(len(Nlist)):
    Results[i]=evaluate_max(M,Nlist[i],p,q)
    print(Nlist[i])
np.save("Results_p055_q1e-3.npy",Results)
#0	\(0\)
#0.1	\(1.23744\times10^{-7}\)
#0.5	\(3.09360\times10^{-6}\)
#2	4.94977×10−5