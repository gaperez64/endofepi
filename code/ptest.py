#!/usr/bin/env python3

"""
Testing:
1. Whether the computed probabilities match the empirical ones via the
   chain binomial simulation,
2. Whether the expected duration of the epidemic as computed exactly matches
   the one that can be approximated using the chain binomial simulation
"""

import math

from endofepi import Model

# Parameter values for tests
h = 1 / 24
GAMMA = 1 / 6
BETA = 1 / 2
REPS = 10000
N = 10  # max pop size

# Instantiate the model
sir = Model(popSize=N,
            h=h,
            beta=BETA,
            gamma=GAMMA)


# Define test suites
def testSucc():
    print(" ++ Testing successor probability values ++ ")

    # Simulate the model from some state a few times
    m = (3, 4, 3)
    succ = dict()
    for _ in range(REPS):
        n = sir.sim(m)
        if n in succ:
            succ[n] += 1
        else:
            succ[n] = 1

    # Print differences with respect to exact values
    for n in succ:
        print("Reached " + str(n) + " " + str(succ[n] / REPS) +
              " vs. Theory probability = " + str(sir.P(m, n)))


def testEoP():
    print(" ++ Testing end of pandemic values ++ ")

    sir.prepAllP()
    eop = sir.endOfPandemic()
    print("Exact mean end of pandemic time = " + str(eop[(N - 1, 1, 0)]))

    # Just to check, the eop dictionary should satisfy the equations from
    # Norris' book
    for m1 in range(0, N + 1):
        for m2 in range(0, N + 1 - m1):
            m3 = N - m1 - m2
            m = (m1, m2, m3)
            assert sum(m) == N
            if m2 == 0:
                assert eop[m] == 0
                continue
            s = 1
            for n1 in range(0, m1 + 1):
                for n3 in range(m3, m3 + m2 + 1):
                    n2 = N - n1 - n3
                    n = (n1, n2, n3)
                    assert sum(n) == N
                    s += sir.P(m, n) * eop[n]
            assert math.isclose(s, eop[m]), "Oh no, miscalc'd value " +\
                                            str(s) + " vs. " + str(eop[m]) +\
                                            " of " + str(m)

    # Simulate pandemic until end for some number of times
    stats = []
    TOTREPS = 0
    while True:
        TOTREPS += REPS
        for _ in range(REPS):
            (m1, m2, m3) = (N - 1, 1, 0)
            steps = 0
            while m2 > 0:
                m = (m1, m2, m3)
                (m1, m2, m3) = sir.sim(m)
                steps += 1
            # print(str((m1, m2, m3)) + " after " + str(steps) + " steps")
            # end of pandemic, store steps
            stats.append(steps)
        # Print mean end of pandemic and diff wrt to exact value
        print("Current empirical mean end of pandemic time = " +
              str(sum(stats) / TOTREPS))
        input("Press Enter to compute some more")


# Run tests
testSucc()
testEoP()
