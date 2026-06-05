import numpy as np

from ga.variation.base import VariationStrategy
from ga.individual import Individual
from ga.variation.mutationClasses import SinglePointMutation, SingePointMutationWithEqualOpportunity, SinglePointWanderMutation, SinglePointWanderMutationWithProbabilityOfNormalMutation, SinglePointTeleportationMutation
from ga.variation.crossoverClasses import SingleBreakCrossover, SingleGridCrossover, GridCrossover, PositiveGeneExchange, PositiveGridGeneExchange


class BasicMutationVariation(VariationStrategy):

    """
    Simple, only mutation
    """
    def __init__(self, rng, ga_config):
        super().__init__(rng, ga_config)

        ms = ga_config.mutation_strategy #{wandering, mutable_wandering, single_point, single_point_equal_opportunity}

        if ms == "wandering":
            self.mutator = SinglePointWanderMutation(ga_config,rng,sigma=ga_config.wandering_mutation_sigma)
        elif ms == "mutable_wandering":
            self.mutator = SinglePointWanderMutationWithProbabilityOfNormalMutation(ga_config,rng,sigma=ga_config.wandering_mutation_sigma)
        elif ms == "single_point":
            self.mutator = SinglePointMutation(ga_config=ga_config,rng=rng)
        elif ms == "single_point_equal_opportunity":
            self.mutator = SingePointMutationWithEqualOpportunity(ga_config=ga_config,rng=rng)
        elif ms == "teleport":
            self.mutator = SinglePointTeleportationMutation(ga_config=ga_config,rng=rng)
        else:
            raise ValueError(f"{ms} is not a valid mutation strategy")

    def variate(
        self, 
        parents : list[Individual],
    ) -> None:
        
        for individual in parents:
            self.mutator.mutate(individual)
        pass

class BasicCrossoverVariation(VariationStrategy):

    """
    Simple, only mutation
    """
    def __init__(self, rng, ga_config):
        super().__init__(rng, ga_config)

        ms = ga_config.crossover_strategy #{single_grid, grid, single_break}

        if ms == "single_grid":
            self.crosser = SingleGridCrossover(ga_config=ga_config, rng=rng)
        elif ms == "grid":
            self.crosser = GridCrossover(ga_config=ga_config, rng=rng, n_pivots=ga_config.n_crossovers)
        elif ms == "single_break":
            self.crosser = SingleBreakCrossover(ga_config=ga_config, rng=rng)
        elif ms == "positive_gene_exchange":
            self.crosser = PositiveGeneExchange(ga_config=ga_config,rng=rng,p_exchange=self.config.p_exchange)
        elif ms == "positive_grid":
            self.crosser = PositiveGridGeneExchange(ga_config=ga_config,rng=rng)
        else:
            raise ValueError(f"{ms} is not a valid crossover strategy")

    def variate(
        self, 
        parents : list[Individual],
    ) -> None:
        
         #last parent if odd will not be crossed and doesn't loose fitness
        
        for i in range(0, len(parents)-1, 2):
            self.crosser.crossover(parent1=parents[i],parent2=parents[i+1]) #+ maybe randomize ?
        pass