from ga.variation.base import CrossoverStrategy
from ga.individual import Individual


class BasicVariation(CrossoverStrategy):

    def variate(
        self, 
        parents : list[Individual]
    ) -> list[Individual]:
        pass