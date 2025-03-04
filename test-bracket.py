from algorithms import *

import numpy
from trotter import Permutations

# Clalibus 5 of 37 =
# a priori chance of streaking each tournament: 0.9473430008563
# probability of winning each match: 0.997427407466306
# Elo Delta: 1035
# Brennon 4 of 50
# a priori chance of streaking each tournament: 0.950740073544508
# probability of winning each match: 0.997597434938851
# Elo Delta: 1047
# Jayme 7 of 22
# a priori chance of streaking each tournament: 0.949280007683847
# probability of winning each match: 0.997524427943751
# Elo Delta: 1042
# Everyone gets 1 rule: (10/11)**(1/10) = 0.990514258214522, elo delta = 808
# pr = 1 / (1 + 10 ** (-delta / 400))
# (1 + 10 ** (-delta / 400)) = 1 / pr
# 10 ** (-delta / 400) = -1 + (1 / pr)
# delta = -400 * log(10, -1 + (1 / pr))
#

items = list(random.sample(range(FLOOR, CEIL), 8))
items = list(numpy.random.normal(AVERAGE, SIGMA, 8))

combos = Permutations(8, items)

elo1 = items[0]
elo2 = items[1]

fixed_gen = [resolve_single_elim(list(range(1,17)), discrete_compare, no_seeding)]
discrete_permutations = DiscretePermutations(16)
discrete_seeded = DiscretePermutationsPerfectSeeding(16)

monte_fixed = monte_carlo(fixed_gen)
print("Monte pure transitive, seeded:")
print(monte_fixed)

monte_dp = monte_carlo(discrete_permutations)
print("Discrete transitive, all permutations:")
print(monte_dp)

monte_ds = monte_carlo(discrete_seeded)
print("Discrete transitive, all permutations, now with seeding!:")
print(monte_ds)

print(eloprobability(AVERAGE + HALF_SIGMA, AVERAGE))
print("First to 10 (+100): 10:" + str(round(10/eloprobability(AVERAGE + HALF_SIGMA, AVERAGE)) - 10))
print("First to 100 (+100): 100:" + str(round(100/eloprobability(AVERAGE + HALF_SIGMA, AVERAGE)) - 100))
print(eloprobability(AVERAGE + SIGMA, AVERAGE))
print("First to 10 (+200): 10:" + str(round(10/eloprobability(AVERAGE + SIGMA, AVERAGE)) - 10))
print("First to 100 (+200): 100:" + str(round(100/eloprobability(AVERAGE + SIGMA, AVERAGE)) - 100))
print(eloprobability(AVERAGE + 2 * SIGMA, AVERAGE))
print("First to 10 (+400): 10:" + str(round(10/eloprobability(AVERAGE + 2 * SIGMA, AVERAGE)) - 10))
print("First to 100 (+400): 100:" + str(round(100/eloprobability(AVERAGE + 2 * SIGMA, AVERAGE)) - 100))
print(eloprobability(CEIL, AVERAGE))
print("First to 10 (+600): 10:" + str(round(10/eloprobability(CEIL, AVERAGE)) - 10))
print("First to 100 (+600): 100:" + str(round(100/eloprobability(CEIL, AVERAGE)) - 100))

print("Minimum Warlord Elo delta: " + str(inv_eloprobability(0.954545454545455)))

elo_probability = EloProbabilityUniform(8, 1, 10)
#print("EloProbabilityUniform(8,1,1): " + str(elo_probability.__next__()))
#print(monte_carlo(elo_probability))

elo_probability = EloProbabilityNormal(8, 1, 10)
#print("EloProbabilityNormal(8,1,1): " + str(elo_probability.__next__()))
#print(monte_carlo(elo_probability))

elo_probability = EloProbabilityNormalWithWarlord(8, 1, 10)
#print("EloProbabilityNormalWithWarlord(8,1,1): " + str(elo_probability.__next__()))
#print(monte_carlo(elo_probability))

brackets = 100
iterations = 100
tests = 30
mean_precision = 3
stddev_precision = 3
for bracket_size in [4, 8, 16]:
    print("\nBracket Size " + str(bracket_size) + "\n")
    print("Monte Elo comparator, uniform distribution, bracket size " + str(bracket_size))
    monte_epu = []
    for i in range(1, tests + 1):
        elo_probability = EloProbabilityUniform(bracket_size, brackets, iterations)
        monte_epu.append(monte_carlo(elo_probability))

    epu_T = [list(i) for i in zip(*monte_epu)]
    for idx, places in enumerate(epu_T):
        print("Mean (" + str(idx + 1) + "P): " + str(round(statistics.mean(places),mean_precision)) + ", Std Dev: " + str(round(statistics.stdev(places),stddev_precision)))

    print("Monte Elo comparator, normal distribution, bracket size " + str(bracket_size))
    monte_epn = []
    for i in range(1, tests + 1):
        elo_probability = EloProbabilityNormal(bracket_size, brackets, iterations)
        monte_epn.append(monte_carlo(elo_probability))

    epn_T = [list(i) for i in zip(*monte_epn)]
    for idx, places in enumerate(epn_T):
        print("Mean (" + str(idx + 1) + "P): " + str(round(statistics.mean(places),mean_precision)) + ", Std Dev: " + str(round(statistics.stdev(places),stddev_precision)))

    print("Monte Elo comparator, normal distribution, perfect seeding, bracket size " + str(bracket_size))
    monte_epp = []
    for i in range(1, tests + 1):
        elo_probability = EloProbabilityNormalPerfectSeeding(bracket_size, brackets, iterations)
        monte_epp.append(monte_carlo(elo_probability))

    epp_T = [list(i) for i in zip(*monte_epp)]
    for idx, places in enumerate(epp_T):
        print("Mean (" + str(idx + 1) + "P): " + str(round(statistics.mean(places),mean_precision)) + ", Std Dev: " + str(round(statistics.stdev(places),stddev_precision)))

    print("Monte Elo comparator, normal distribution + Warlord, bracket size " + str(bracket_size))
    monte_epw = []
    for i in range(1, tests + 1):
        elo_probability = EloProbabilityNormalWithWarlord(bracket_size, brackets, iterations)
        monte_epw.append(monte_carlo(elo_probability))

    epw_T = [list(i) for i in zip(*monte_epw)]
    for idx, places in enumerate(epw_T):
        print("Mean (" + str(idx + 1) + "P): " + str(round(statistics.mean(places),mean_precision)) + ", Std Dev: " + str(round(statistics.stdev(places),stddev_precision)))


    print("Monte Elo comparator, normal distribution + Warlord, perfect seeding, bracket size " + str(bracket_size))
    monte_epi = []
    for i in range(1, tests + 1):
        elo_probability = EloProbabilityNormalWithWarlordPerfectSeeding(bracket_size, brackets, iterations)
        monte_epi.append(monte_carlo(elo_probability))

    epi_T = [list(i) for i in zip(*monte_epi)]
    for idx, places in enumerate(epi_T):
        print("Mean (" + str(idx + 1) + "P): " + str(round(statistics.mean(places),mean_precision)) + ", Std Dev: " + str(round(statistics.stdev(places),stddev_precision)))
