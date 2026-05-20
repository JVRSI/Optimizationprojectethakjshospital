import numpy as np


from ga.variation.base import VariationStrategy
from ga.individual import Individual
from ga.variation.basic import BasicCrossoverVariation

class MicroGAVariation(VariationStrategy):

    def __init__(self, rng, ga_config):
        super().__init__(rng, ga_config)

        self.crosser = BasicCrossoverVariation(rng=rng,ga_config=ga_config)

    def variate(
        self, 
        parents : list[Individual],
    ):
        """
        To be used with small population size and number of parents only slightly lower then population size

        population_size - n_parents = number of elites kept

        precondition:
        population_size > n_parents

        optionally but good:
        n_parents % 2 == 0

        """

        n_elite = self.config.population_size - self.config.n_parents

        fitness = np.fromiter((p.fitness for p in parents), dtype=float)  #!COMPUTATION, might be to much overhead

        idx = np.argpartition(fitness, n_elite)[:n_elite]
        elite = [parents[i].clone() for i in idx]   #elite gets "deep copy", rest gets in place crossover (including old references to elite)

        idx = self.rng.choice(len(parents), size=self.config.n_parents, replace=False)

        self.crosser.variate(parents=parents[idx])

        parents.extend(elite)