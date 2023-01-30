#!/usr/bin/env python3

"""
End of Epidemic computes the expected end of the epidemic
for a discrete-time stochastic epidemic model

authors: Guillermo A. Perez, guillermo.perez@uantwerpen.be
         Alejandro Alarcon, alejandrojavier.alarcongonzalez@uantwerpen.be
"""

import argparse
import numpy as np
import scipy.special as sp


class Model:
    def __init__(self, popSize, h, beta, gamma):
        self.popSize = popSize
        self.h = h
        self.beta = beta
        self.gamma = gamma
        self.exphgamma = np.exp(-h * gamma)
        self.compExphgamma = 1 - self.exphgamma
        self.knownP = dict()

    def sim(self, m):
        (m1, m2, m3) = m
        varlambda = self.beta * m2
        pStar = 1 - np.exp(-self.h * varlambda)
        Inew = np.random.binomial(m1, pStar)
        Rnew = np.random.binomial(m2, self.compExphgamma)

        # Compartmental updating rules
        n1 = m1 - Inew
        n2 = m2 + Inew - Rnew
        n3 = m3 + Rnew
        n = (n1, n2, n3)
        assert sum(n) == self.popSize
        return n

    def P(self, m, n):
        if (m, n) in self.knownP:
            p = self.knownP[(m, n)]
        else:
            (m1, m2, m3) = m
            (n1, n2, n3) = n
            p = sp.binom(m1, m1 - n1)
            p *= np.exp(-self.h * self.beta * m2 * n1)
            if m1 - n1 > 0:
                p *= np.power(1 - np.exp(-self.h * self.beta * m2), m1 - n1)
            p *= sp.binom(m2, n3 - m3)
            p *= np.power(self.compExphgamma, n3 - m3)
            p *= np.power(self.exphgamma, m2 - n3 + m3)
            self.knownP[(m, n)] = p
        return p

    def prepAllP(self):
        def alpha(m, n):
            (m1, m2, m3) = m
            (n1, n2, n3) = n
            a = m1
            a *= m2 - n2 + m3 + 1
            a /= n3 - m3
            a *= self.compExphgamma
            a /= m1 - n1
            a *= 1 - np.exp(-self.h * self.beta * m2)
            a /= self.exphgamma
            return a

        # initialization
        for m1 in range(self.popSize + 1):
            m = (m1, 0, self.popSize - m1)
            self.knownP[(m, m)] = 1

        # compute the rest of the probabilities
        for M in range(1, self.popSize + 1):
            for m2 in range(1, M + 1):
                m1 = M - m2
                m3 = self.popSize - M
                m = (m1, m2, m3)
                assert sum(m) == self.popSize
                # The case of n1 = m1
                for n3 in range(m3, m2 + m3 + 1):
                    n = (m1, self.popSize - m1 - n3, n3)
                    self.P(m, n)
                # The case of n3 = m3
                for n1 in range(0, m1 + 1):
                    n = (n1, self.popSize - m3 - n1, m3)
                    assert sum(n) == self.popSize
                    self.P(m, n)
                # All other cases
                for n3 in range(m3 + 1, m2 + m3 + 1):
                    for n1 in range(0, m1):
                        n = (n1, self.popSize - n1 - n3, n3)
                        mm = (m1 - 1, m2, m3 + 1)
                        assert sum(mm) == self.popSize
                        self.knownP[(m, n)] = alpha(m, n) * \
                            self.knownP[(mm, n)]

    def endOfPandemic(self):
        val = dict()
        # initialization
        for m1 in range(self.popSize + 1):
            m = (m1, 0, self.popSize - m1)
            assert sum(m) == self.popSize
            val[m] = 0
        for M in range(1, self.popSize + 1):
            for m2 in reversed(range(1, M + 1)):
                m1 = M - m2
                m3 = self.popSize - M
                m = (m1, m2, m3)
                assert sum(m) == self.popSize
                val[m] = 1
                for n3 in range(m3, m2 + m3 + 1):
                    for n1 in range(m1 + 1):
                        n2 = self.popSize - (n1 + n3)
                        n = (n1, n2, n3)
                        assert sum(n) == self.popSize
                        if m != n:
                            val[m] += self.P(m, n) * val[n]
                # normalize because of self loops
                val[m] /= (1 - self.P(m, m))
        # return the full dictionary
        return val


# Set up argument parsing for this file when called as a script instead of a
# library
parser = argparse.ArgumentParser(
    prog="endofepi",
    description="This program computes the expected end of "
                "epidemic time for a discrete-time stochastic "
                "epidemic model")
parser.add_argument("population_size", type=int)
parser.add_argument("time_step", type=float)
parser.add_argument("infection_rate", type=float)
parser.add_argument("recovery_rate", type=float)

if __name__ == "__main__":
    args = parser.parse_args()
    m = Model(popSize=args.population_size,
              h=args.time_step,
              beta=args.infection_rate,
              gamma=args.recovery_rate)
    m.prepAllP()
    val = m.endOfPandemic()
    print(val[(args.population_size - 1, 1, 0)])
    exit(0)
