import random
import statistics
import math

import numpy
from trotter import Permutations

AVERAGE = 1500
SIGMA = 200
HALF_SIGMA = SIGMA / 100
SIGMA3 = 3 * SIGMA
FLOOR = AVERAGE - SIGMA3
CEIL = AVERAGE + SIGMA3

def inv_eloprobability(pr):
    return round(-400 * math.log(-1 + (1 / pr), 10))

def eloprobability(elo1, elo2):
    pr_elo1wins = 1 / (1 + 10 ** (-(elo1 - elo2) / 400))
    return pr_elo1wins

def elo_compare(elo1, elo2):
    pr_elo1wins = eloprobability(elo1, elo2)
    cond = random.uniform(0, 1)
    return 1 if cond < pr_elo1wins else -1

def discrete_compare(elo1, elo2):
    return elo1 - elo2

def resolve_single_elim_round(scores, comparator, seeding):
    scores = seeding(scores)
    A = scores[:len(scores)//2]
    B = scores[len(scores)//2:]
    return list(map(lambda x, y: x if comparator(x, y) > 0 else y, A, B))

def no_seeding(scores):
    return scores

def perfect_seeding(scores):
    scores = list(reversed(sorted(scores)))
    scores = scores[0:len(scores)//2] + list(reversed(scores[len(scores)//2:]))
    return scores

def resolve_single_elim(scores, comparator, seeding):
    if len(scores) == 1:
        return scores[0]
    rounds = [scores]
    S = resolve_single_elim_round(scores, comparator, seeding)
    rounds.append(S)
    while len(S) > 1:
        S = resolve_single_elim_round(S, comparator, seeding)
        rounds.append(S)
    if len(rounds) > 1:
        results = S + list(set(rounds[-2]) - set(S))
    if len(rounds) > 2:
        results += [list(sorted(list(set(rounds[-3]) - set(results))))[1]]
    return results

def monte_carlo(bracket_results_gen):
    total = 0
    pval = [0, 0, 0]
    for results in bracket_results_gen:
        ideal = list(reversed(sorted(results)))[0:3]
        total += 1
        for idx in range(min(len(results), 3)):
            if results[idx] == ideal[idx]:
                pval[idx] += 1
    return [p/total for p in pval]

class DiscretePermutations:
    def __init__(self, bracket_size):
        self.bracket_size = bracket_size
        items = list(range(1,9))
        self.combos = Permutations(8, items)
        self.idx = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.combos)

    def __next__(self):
        if self.idx >= len(self.combos):
            raise StopIteration
        next = self.resolve_bracket()
        self.idx += 1
        return next

    def resolve_bracket(self):
        return resolve_single_elim(self.combos[self.idx], discrete_compare, no_seeding)

class DiscretePermutationsPerfectSeeding(DiscretePermutations):
    def resolve_bracket(self):
        return resolve_single_elim(self.combos[self.idx], discrete_compare, perfect_seeding)


class EloProbabilityUniform:
    def __init__(self, bracket_size, brackets, iterations):
        self.bracket_size = bracket_size
        self.bracket_count = brackets
        self.bracket_iterations = iterations
        self.current_bracket = 0
        self.current_bracket_iteration = 0
        self.bracket = self.get_bracket()

    def __iter__(self):
        return self

    def __len__(self):
        return self.bracket_count * self.bracket_iterations

    def __next__(self):
        if self.current_bracket_iteration >= self.bracket_iterations:
            self.current_bracket_iteration = 0
            self.current_bracket += 1
            self.bracket = self.get_bracket()
        self.current_bracket_iteration += 1
        if self.current_bracket_iteration >= self.bracket_iterations and self.current_bracket >= self.bracket_count:
            raise StopIteration
        return self.resolve_bracket()

    def resolve_bracket(self):
        return resolve_single_elim(self.bracket, elo_compare, no_seeding)

    def get_bracket(self):
        bracket = list(random.sample(range(FLOOR, CEIL), self.bracket_size))
        #print("uniform bracket: " + str(bracket))
        return bracket

class EloProbabilityNormal(EloProbabilityUniform):
    def get_bracket(self):
        bracket = list(numpy.random.normal(AVERAGE, SIGMA, self.bracket_size))
        #print("normal bracket: " + str(bracket))
        return bracket

class EloProbabilityNormalPerfectSeeding(EloProbabilityUniform):
    def get_bracket(self):
        bracket = list(numpy.random.normal(AVERAGE, SIGMA, self.bracket_size))
        #print("perfect bracket: " + str(bracket))
        return bracket

    def resolve_bracket(self):
        return resolve_single_elim(self.bracket, elo_compare, perfect_seeding)

class EloProbabilityNormalWithWarlord(EloProbabilityUniform):
    def get_bracket(self):
        scores = list(numpy.random.normal(AVERAGE, SIGMA, self.bracket_size))
        scores.remove(max(scores))
        scores.append(400 + max(scores))
        #print("warlord bracket: " + str(scores))
        return scores

class EloProbabilityNormalWithWarlordPerfectSeeding(EloProbabilityNormalPerfectSeeding):
    def get_bracket(self):
        scores = list(numpy.random.normal(AVERAGE, SIGMA, self.bracket_size))
        scores.remove(max(scores))
        scores.append(400 + max(scores))
        #print("warlord bracket: " + str(scores))
        return scores
