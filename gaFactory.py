import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from pathlib import Path
from dataclasses import asdict, dataclass
import json
import sys
import argparse
from typing import Callable


from paths import DATA_DIR, RUNS_DIR, MATRIX_PATH, RUNS_LOCAL

from ga import GAConfig, GeneticAlgorithm, BasicGenerator, GravityGenerator, ParallelEvaluator
from ga.genome_generator import GenomeGenerator
from ga.selection import RouletteSelection, SelectionStrategy
from ga.variation import EvolutionaryVariation, MicroGAVariation, ClassicVariation, VariationStrategy
from ga.evaluator import Evaluator
from ga.analysis.statistics import GAStatistics

from testSim import matrix_to_city_dataframe, load_population_matrix



from Simulation import Simulation, SimConfig
from Simulation.entities import City

@dataclass
class GAFactory:
    ga_config: GAConfig
    generator_factory: Callable[[GAConfig,int], object]
    selector_factory: Callable[[int,int], object]
    variator_factory: Callable[[GAConfig,int], object]
    generator_name: str
    selector_name: str
    variator_name: str
    idea: str

    def __str__(self):
        return (
            f"Generator: {self.generator_name}\n"
            f"Selection: {self.selector_name}\n"
            f"Variation: {self.variator_name}\n"
            f"Idea: {self.idea}"
        )

