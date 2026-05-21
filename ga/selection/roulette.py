import numpy as np


from ga.selection.base import SelectionStrategy
from ga.individual import Individual
from ga.population import Population


class RouletteSelection(SelectionStrategy):

    def select(
            self,
            population : Population,
        ):

        #use  f_max - f_i + epsilon  to build roulette (currently fitness is between 0 and 1, might want to change that)
        f_max = population.worst().fitness    #!COMPUTATION, finding worst might be inefficient -> change to 1 / fitness
        epsilon = 1e-6

        cumsum = np.cumsum(
            f_max - np.fromiter((i.fitness for i in population), dtype=float) + epsilon
        )

        r = self.rng.random(self.n_parents) * cumsum[-1]

        

        indices = np.searchsorted(cumsum,r)

        population.individuals =  [population.individuals[i] for i in indices]