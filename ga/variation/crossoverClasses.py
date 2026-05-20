from abc import ABC, abstractmethod
import numpy as np
#from typing import Annotated
from typing import TypeAlias
from bisect import bisect_right


from ga.individual import Individual
from ga.config import GAConfig


Gene: TypeAlias = tuple[int, int, int]
Genome: TypeAlias = list[Gene]

class CrossoverStrategy(ABC):
    def __init__(
            self,
            ga_config : GAConfig,
            rng : np.random.Generator,
        ):
        super().__init__()
        self.height = ga_config.genome_size[0]
        self.width = ga_config.genome_size[1]
        self.n_genes = self.height * self.width
        self.rng = rng

    @abstractmethod
    def crossover(
        self,
        parent1 : Individual,
        parent2 : Individual,
    ) -> None:  
        pass

    def get_position_from_total_index(self, idx): #total index as in the index of the city when enumerating all cities
        row = idx // self.width
        col = idx % self.height
        return row, col
    


class SingleBreakCrossover(CrossoverStrategy):

    def crossover(
            self, 
            parent1 : Individual, 
            parent2 : Individual, 
        ): 

        pivot = (self.get_position_from_total_index(self.rng.integers(self.n_genes)))


        move_1_to_2 = [t for t in parent1.genome if (t[1], t[2]) > pivot]
        keep_1      = [t for t in parent1.genome if (t[1], t[2]) <= pivot]

        move_2_to_1 = [t for t in parent2.genome if (t[1], t[2]) <= pivot]
        keep_2      = [t for t in parent2.genome if (t[1], t[2]) > pivot]

        parent1.genome = keep_1 + move_2_to_1
        parent2.genome = keep_2 + move_1_to_2


class SingleGridCrossover(CrossoverStrategy):
    def crossover(
            self, 
            parent1 : Individual, 
            parent2 : Individual, 
        ): 
        """
        Idea:

        xxxxxx|ooo
        xxxxxx|ooo
        xxxxxx|ooo
        ----------
        oooooo|xxx
        oooooo|xxx

        """

        r,c = (self.get_position_from_total_index(self.rng.integers(self.n_genes)))


        move_1_to_2 = [t for t in parent1.genome if not ((t[1] < r) == (t[2] < c))]
        keep_1      = [t for t in parent1.genome if     ((t[1] < r) == (t[2] < c))]

        move_2_to_1 = [t for t in parent2.genome if     ((t[1] < r) == (t[2] < c))]
        keep_2      = [t for t in parent2.genome if not ((t[1] < r) == (t[2] < c))]

        parent1.genome = keep_1 + move_2_to_1
        parent2.genome = keep_2 + move_1_to_2



class GridCrossover(CrossoverStrategy):
    
    def __init__(
            self,
            ga_config : GAConfig,
            rng : np.random.Generator,
            n_pivots : int = 10
        ):
        super().__init__(ga_config, rng)
        self.n_pivots = n_pivots

    def crossover(
            self, 
            parent1 : Individual, 
            parent2 : Individual, 
        ): 
        row_pivots = sorted(self.rng.choice(self.height, size=self.n_pivots, replace=False))
        col_pivots = sorted(self.rng.choice(self.height, size=self.n_pivots, replace=False))

        parent1.genome = (
            [t for t in parent1.genome if self.from_parent1(t,row_pivots,col_pivots)]
            +
            [t for t in parent2.genome if not self.from_parent1(t,row_pivots,col_pivots)]
        )

        parent2.genome = (
            [t for t in parent2.genome if self.from_parent1(t,row_pivots,col_pivots)]
            +
            [t for t in parent1.genome if not self.from_parent1(t,row_pivots,col_pivots)]
        )
    

    def from_parent1(self,t,row_pivots,col_pivots):
        row, col = t[1], t[2]

        rr = bisect_right(row_pivots, row)
        cc = bisect_right(col_pivots, col)

        return (rr + cc) % 2 == 0