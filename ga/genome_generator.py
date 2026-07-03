from abc import ABC, abstractmethod
from typing import TypeAlias
import numpy as np
from pathlib import Path
import json


from ga.config import GAConfig

Gene: TypeAlias = tuple[int, int, int]
Genome: TypeAlias = list[Gene]


class GenomeGenerator(ABC):

    def __init__(
        self, 
        rng: np.random.Generator,
        config : GAConfig,
    ):
        self.rng = rng
        self.config = config

        if self.config.random_amount:
            large = self.rng.normal(
                loc=self.config.mean_hospital_large,
                scale=1
            )
            small = self.rng.normal(
                loc=self.config.mean_hospital_small,
                scale=3
            )
        else:
            large = self.config.mean_hospital_large
            small = self.config.mean_hospital_small

        self.m = int(round(max(1,large)))
        self.n = int(round(max(2,large + small)))

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

        cities_matrix = cities_matrix**0.7

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
    
class LoadFromFile(GenomeGenerator):
    def __init__(
            self, 
            rng : np.random.Generator, 
            config : GAConfig,
            folder_path : Path,
            cities_matrix
        ):
        super().__init__(rng, config)

        self.file_path = folder_path / "last_generation.json"
        if not self.file_path.exists():
            raise FileNotFoundError(f"File {self.file_path} doesn't exist")

        with self.file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.individuals = data["individuals"]
        self.i = 0
        self.from_file = True

        self.rrs = GravityGenerator(rng=rng,config=config,cities_matrix=cities_matrix)


    
    def __call__(self) -> Genome | None:
        if self.i >= len(self.individuals) and self.from_file:
            self.from_file = False
            return None

        if self.from_file:
            genome_data = self.individuals[self.i]["genome"]
            self.i += 1

            # [[type, row, col], ...] -> [(type, row, col), ...]
            return [tuple(gene) for gene in genome_data]
        else:
            return self.rrs()

