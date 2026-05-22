import numpy as np

from ga.variation.base import VariationStrategy
from ga.individual import Individual
from ga.variation.basic import BasicMutationVariation, BasicCrossoverVariation

class ClassicVariation(VariationStrategy):

    def __init__(self, rng, ga_config):
        super().__init__(rng, ga_config)

        self.mutator = BasicMutationVariation(rng=rng,ga_config=ga_config)
        self.crosser = BasicCrossoverVariation(rng=rng,ga_config=ga_config)
        self.pm = ga_config.probability_of_mutation
        self.pc = ga_config.probability_of_crossover
        self.n = ga_config.population_size
        self.k = ga_config.n_parents
        self.m = self.n - self.k

    
    def variate(
        self, 
        parents : list[Individual],
    ):
        """
        n_parents should be reasonably smaller than population_size
        """
        # input K parents from selection
        
        # select M = N-K parents to duplicate (-K because of computation, we already have them so why not keep) #!LOGIC ?
        idx = self.rng.choice(self.k, size=self.m, replace=True)
        parents.extend([parents[i].clone() for i in idx])

        #select random individuals for crossover
        nc = self.rng.binomial(self.n,self.pc)
        idx = self.rng.choice(self.n, size=nc, replace=False)
        self.crosser.variate([parents[i] for i in idx])

        #select random individuals for mutation
        nm = self.rng.binomial(self.n,self.pm)
        idx = self.rng.choice(self.n, size=nm, replace=False)
        self.mutator.variate([parents[i] for i in idx])
