#!/usr/bin/env python3

"""
Testing:
1. Whether the computed probabilities match the empirical ones via the
   chain binomial simulation,
2. Whether the expected duration of the epidemic as computed exactly matches
   the one that can be approximated using the chain binomial simulation
"""

from endofepi import Model

# Parameter values for tests
h = 1 / 24
GAMMA = 1 / 6
BETA = 1 / 2
# Max pop size
N = 30

# Instantiate the model
sir = Model(popSize=N,
            h=h,
            beta=BETA,
            gamma=GAMMA)

# Simulate the model from some state a few times
m = (10, 10, 10)
succ = dict()
NEXP = 100000
for _ in range(NEXP):
    n = sir.sim(m)
    if n in succ:
        succ[n] += 1
    else:
        succ[n] = 1
print(succ)

# Print differences with respect to exact values
for n in succ:
    print("Reached " + str(n) + " " + str(succ[n] / NEXP) +
          " vs. Theory probability = " + str(sir.P(m, n)))
