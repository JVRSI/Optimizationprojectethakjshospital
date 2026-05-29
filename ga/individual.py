from __future__ import annotations #to use -> Individual already in the Individual class, also need to change Generator
from typing import TypeAlias
from dataclasses import asdict
import numpy as np

Gene: TypeAlias = tuple[int, int, int]
Genome: TypeAlias = list[Gene]  # if changed to contain mutable (List, dict, etc.) within List we need to change self.copy() to deepcopy

class Individual:

    def __init__(
            self, 
            genome : Genome 
        ):
        #if not isinstance(genome, Genome):
        #    raise TypeError(f"config must be instance of GAConfig, got {type(genome)}")
        
        self.genome = genome
        self.fitness = None
        self.sim_records = None

    def clone(self) -> Individual:
        clone = Individual(self.genome.copy())  #copy is enough as List only contains immutables (tuples), if it would contain List or Dic we would need to deepcopy
        clone.fitness = self.fitness
        clone.sim_records = self.sim_records
        return clone

    # on print(individual) this string is returned
    def __repr__(self):
        return f"genome: {self.genome}\nfitness: {self.fitness}"
    
    def to_dict(self):
        return {
            "genome": [
                [int(gene[0]), int(gene[1]), int(gene[2])]  # tuple -> list
                for gene in self.genome
            ],
            "fitness": self.fitness,
            "sim_records": (
                asdict(self.sim_records)
                if self.sim_records is not None
                else None
            )
        }
    
    def convert_numpy(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self.convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.convert_numpy(v) for v in obj]
        return obj