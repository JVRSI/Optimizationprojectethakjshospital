from ga.population import Population
from ga.individual import Individual
from ga.config import GAConfig
from ga.selection.base import SelectionStrategy
from ga.variation.base import VariationStrategy
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
        variation : VariationStrategy,
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
        if not isinstance(variation, VariationStrategy):
            raise TypeError(f"crossover must be instance of Crossover, got {type(variation)}")
        if not isinstance(replacement, ReplacementStrategy):
            raise TypeError(f"replacement must be instance of Replacement, got {type(replacement)}")
        if not isinstance(evaluator, Evaluator):
            raise TypeError(f"evaluator must be instance of Evaluator, got {type(evaluator)}")
        if not isinstance(rng, np.random.Generator):
            raise TypeError(f"evaluator must be instance of Evaluator, got {type(rng)}")

        self.config = config

        self.selection = selection
        self.variation = variation
        self.replacement = replacement
        self.evaluator = evaluator

        self.rng = rng

        self.population = Population(
            [
                Individual(genome_generator())
                for _ in range(self.config.population_size)
            ]
        )

    def initialize(self) -> None:
        """
        Bereitet den genetischen Algorithmus für den Start vor.

        Typische Aufgaben:
        - Bewertet die initiale Population.
        - Setzt Statistiken oder Tracking-Variablen zurück.
        - Initialisiert Generationenzähler.
        - Optional: sortiert Population nach Fitness.
        """

        # initial evaluation
        self.evaluator.evaluate(
            self.population.individuals
        )
        
        #+ statistics

        pass


    def step(self, generation) -> None:
        """
        Updating one generation
        """

        # select parents to construct offspring from
        parents = self.selection.select(self.population.individuals, self.config.n_parents)

        # create offspring
        offspring = self.variation.variate(parents)

        # evaluate offspring
        self.evaluator.evaluate(offspring)

        # replace old population
        self.population.individuals = self.replacement.replace(self.population.individuals, offspring)

        #+ statistics


    def run(self, generations):
        """
        Führt den genetischen Algorithmus über mehrere Generationen aus.

        Parameter:
        - generations:
            Anzahl der Generationen, die simuliert werden sollen.

        Typische Aufgaben:
        - initialize() einmal aufrufen.
        - Für jede Generation step() ausführen.
        - Optional: Fortschritt loggen oder beste Lösung speichern.
        - Am Ende die beste gefundene Lösung zurückgeben.
        """

        self.initialize()

        for generation in range(generations):
            self.step(generation)
            

        return best_individual
