from typing import TypeAlias
from __future__ import annotations #to use -> Individual already in the Individual class, also need to change Generator

Gene: TypeAlias = tuple[str, tuple[int, int]]
Genome: TypeAlias = list[Gene]  # if changed to contain mutable (List, dict, etc.) within List we need to change self.copy() to deepcopy

class Individual:

    def __init__(
            self, 
            genome : Genome 
        ):
        if not isinstance(genome, Genome):
            raise TypeError(f"config must be instance of GAConfig, got {type(genome)}")
        
        self.genome = genome
        self.fitness = None

    def copy(self) -> Individual:
        clone = Individual(self.genome.copy())  #copy is enough as List only contains immutables, if it would contain List or Dic we would need to deepcopy
        clone.fitness = self.fitness
        return clone

    # on print(individual) this string is returned
    def __repr__(self):
        return f"genome: {self.genome}\nfitness: {self.fitness}"