from abc import ABC, abstractmethod
from typing import TypeAlias
import numpy as np

from ga.config import GAConfig

Gene: TypeAlias = tuple[str, tuple[int, int]]
Genome: TypeAlias = list[Gene]


class Generator(ABC):

    def __init__(
        self, 
        rng: np.random.Generator,
        config : GAConfig
    ):
        self.rng = rng
        self.config = config

    @abstractmethod
    def __call__(self) -> Genome:
        pass


class BasicGenerator(Generator):
    '''

    Returns a valid random Genome of fixed size

    number of large hospitals = config.mean_hospitals_large
    number of small hospitals = config.mean_hospitals_small

    '''

    def __call__(self) -> Genome:
        height = self.config.genome_size[0]
        width = self.config.genome_size[1]
        total = width * height

        n = self.config.mean_hospital_large + self.config.mean_hospital_small
        m = self.config.mean_hospital_large

        if n > total:
            raise ValueError("Total amount of placable hospitals extends total number of squares")

        # random unique indices, not sorted
        idx = self.rng.choice(total, size=n, replace=False)

        # transform to (row,col)
        row = idx // width
        col = idx % width

        # assign type of hospital
        types = np.array(["S"] * n, dtype=object)
        types[:m] = "L"

        return list(zip(types, zip(row, col)))


