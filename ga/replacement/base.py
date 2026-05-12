from abc import ABC, abstractmethod

from ga.individual import Individual

class ReplacementStrategy(ABC):

    @abstractmethod
    def replace(
        self,
        old_population : list[Individual],
        offspring : list[Individual],
    ) -> list[Individual]:
        pass