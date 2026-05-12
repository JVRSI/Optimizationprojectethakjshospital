from abc import ABC, abstractmethod

from ga.individual import Individual

class SelectionStrategy(ABC):

    @abstractmethod
    def select(self, population, n_parents) -> list[Individual]:
        pass