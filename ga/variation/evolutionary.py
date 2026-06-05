import numpy as np

from ga.variation.base import VariationStrategy
from ga.individual import Individual
from ga.variation.basic import BasicMutationVariation

class EvolutionaryVariation(VariationStrategy):

    def __init__(self, rng, ga_config):
        super().__init__(rng, ga_config)

        self.mutator = BasicMutationVariation(rng=rng,ga_config=ga_config)

        self.k = self.config.population_size - self.config.n_parents - self.config.n_elites
        self.m = self.config.n_parents

    def variate(
        self, 
        parents : list[Individual],
    ):
        idx = self.rng.choice(len(parents), size=self.k, replace=True)
        children = [parents[i].clone() for i in idx]   # fitness gets cloned too, but will be deleted when doing mutation
        self.mutator.variate(children)

        parents.extend(children)