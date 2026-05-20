from abc import ABC, abstractmethod
from numpy.random import Generator

from ga.individual import Individual
from ga.config import GAConfig

class VariationStrategy(ABC):

    def __init__(
            self,
            rng : Generator,
            ga_config : GAConfig,
        ):
        super().__init__()
        self.config = ga_config
        self.rng = rng


    @abstractmethod
    def variate(
        self, 
        parents : list[Individual],
    ):
        pass
