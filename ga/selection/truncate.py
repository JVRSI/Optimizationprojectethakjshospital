from ga.selection.base import SelectionStrategy
from numpy.random import Generator

from ga.individual import Individual
from ga.population import Population

class TruncateSelection(SelectionStrategy):

    def __init__(
            self,
            n_parents : int,
            rng : Generator
        ):
        self.n_parents = n_parents

    def select(
            self,
            population : Population,
        ):
        population.sort_population()

        for i in range(population.size()-1,self.n_parents-1,-1):
            del population.individuals[i]