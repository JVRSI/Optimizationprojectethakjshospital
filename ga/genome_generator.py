from abc import ABC, abstractmethod
from typing import TypeAlias
import numpy as np

from ga.config import GAConfig

Gene: TypeAlias = tuple[int, int, int]
Genome: TypeAlias = list[Gene]


class GenomeGenerator(ABC):

    def __init__(
        self, 
        rng: np.random.Generator,
        config : GAConfig
    ):
        self.rng = rng
        self.config = config

        self.n = self.config.mean_hospital_large + self.config.mean_hospital_small
        self.m = self.config.mean_hospital_large

        self.height = self.config.genome_size[0]
        self.width = self.config.genome_size[1]
        self.total = self.width * self.height

    @abstractmethod
    def __call__(self) -> Genome:
        pass

    def get_position_from_total_index(self, idx): 
        row = idx // self.width
        col = idx % self.width
        return row, col


class BasicGenerator(GenomeGenerator):
    '''

    Returns a valid random Genome of fixed size

    number of large hospitals = config.mean_hospitals_large
    number of small hospitals = config.mean_hospitals_small

    '''

    def __call__(self) -> Genome:

        if self.n > self.total:
            raise ValueError("Total amount of placable hospitals extends total number of squares")

        # random unique indices, not sorted
        idx = self.rng.choice(self.total, size=self.n, replace=False)

        # transform to (row,col)
        row, col = self.get_position_from_total_index(idx)

        # assign type of hospital
        types = np.array([1] * self.n)
        types[:self.m] = 2

        return list(zip(types, row, col))    

class GravityGenerator(GenomeGenerator):
    
    def __init__(
            self, 
            rng : np.random.Generator, 
            config : GAConfig,
            cities_matrix : np.ndarray,
        ):
        super().__init__(rng, config)


        epsilon = 1e-6

        #array of accumulated sums
        self.cumsum = np.cumsum(
            np.ravel(cities_matrix) + epsilon
        )

    
    def __call__(self) -> Genome:

        #random indicies with higher chance in squares with more population
        r = self.rng.random(self.n) * self.cumsum[-1]
        i = np.searchsorted(self.cumsum,r)

        # transform to (row,col)
        row, col = self.get_position_from_total_index(i)

        # assign type of hospital
        types = np.array([1] * self.n)
        types[:self.m] = 2


        return list(zip(types, row, col))

