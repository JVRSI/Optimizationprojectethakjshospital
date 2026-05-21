from abc import ABC, abstractmethod
from numpy.random import Generator

from ga.individual import Individual
from ga.population import Population

class SelectionStrategy(ABC):

    def __init__(
            self,
            n_parents : int,
            rng : Generator
        ):
        super().__init__()
        self.n_parents = n_parents
        self.rng = rng




    @abstractmethod
    def select(
            self,
            population : Population,
        ):
        pass