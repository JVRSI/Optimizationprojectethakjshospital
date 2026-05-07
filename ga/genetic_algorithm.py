from ga.population import Population
from ga.individual import Individual
from ga.config import GAConfig
from ga.selection.base import SelectionStrategy
from ga.crossover.base import CrossoverStrategy
from ga.mutation.base import MutationStrategy
from ga.replacement.base import ReplacementStrategy
from ga.evaluator import Evaluator
from ga.generator import Generator

import numpy as np



class GeneticAlgorithm:

    def __init__(
        self,
        config : GAConfig,
        genome_generator : Generator,
        selection : SelectionStrategy,
        crossover : CrossoverStrategy,
        mutation : MutationStrategy,
        replacement : ReplacementStrategy,
        evaluator : Evaluator,
        rng : np.random.Generator,
    ):
        if not isinstance(config, GAConfig):
            raise TypeError(f"config must be instance of GAConfig, got {type(config)}")
        if not isinstance(self.genome_generator, Generator):
            raise TypeError(f"genome_generator must be instance of Generator, got {type(genome_generator)}")
        if not isinstance(selection, SelectionStrategy):
            raise TypeError(f"selection must be instance of Selection, got {type(selection)}")
        if not isinstance(crossover, CrossoverStrategy):
            raise TypeError(f"crossover must be instance of Crossover, got {type(crossover)}")
        if not isinstance(mutation, MutationStrategy):
            raise TypeError(f"mutation must be instance of Mutation, got {type(mutation)}")
        if not isinstance(replacement, ReplacementStrategy):
            raise TypeError(f"replacement must be instance of Replacement, got {type(replacement)}")
        if not isinstance(evaluator, Evaluator):
            raise TypeError(f"evaluator must be instance of Evaluator, got {type(evaluator)}")
        if not isinstance(rng, np.random.Generator):
            raise TypeError(f"evaluator must be instance of Evaluator, got {type(rng)}")

        self.config = config

        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.replacement = replacement
        self.evaluator = evaluator

        self.rng = rng

        self.population = Population(
            [
                Individual(genome_generator())
                for _ in range(self.config.population_size)
            ]
        )

    def initialize(self):
        pass

    def step(self):
        pass

    def run(self, generations):
        pass

    def create_offspring(self, parents):
        pass

    def evaluate_population(self):
        pass