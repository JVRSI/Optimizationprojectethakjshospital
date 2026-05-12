from abc import ABC, abstractmethod

from ga.individual import Individual

class VariationStrategy(ABC):

    @abstractmethod
    def variate(
        self, 
        parents : list[Individual]
    ) -> list[Individual]:
        pass