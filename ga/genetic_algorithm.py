from ga.population import Population
from ga.individual import Individual
from ga.config import GAConfig
from ga.selection.base import SelectionStrategy
from ga.variation.base import VariationStrategy
from ga.evaluator import Evaluator
from ga.genome_generator import GenomeGenerator
from ga.analysis.statistics import GAStatistics

import numpy as np





class GeneticAlgorithm:

    def __init__(
        self,
        config : GAConfig,
        genome_generator : GenomeGenerator,
        selection : SelectionStrategy,
        variation : VariationStrategy,
        evaluator : Evaluator,
        statistics : GAStatistics,
        rng : np.random.Generator,
    ):
        if not isinstance(config, GAConfig):
            raise TypeError(f"config must be instance of GAConfig, got {type(config)}")
        if not isinstance(genome_generator, GenomeGenerator):
            raise TypeError(f"genome_generator must be instance of GenomeGenerator, got {type(genome_generator)}")
        if not isinstance(selection, SelectionStrategy):
            raise TypeError(f"selection must be instance of Selection, got {type(selection)}")
        if not isinstance(variation, VariationStrategy):
            raise TypeError(f"crossover must be instance of Crossover, got {type(variation)}")
        if not isinstance(evaluator, Evaluator):
            raise TypeError(f"evaluator must be instance of Evaluator, got {type(evaluator)}")
        if not isinstance(rng, np.random.Generator):
            raise TypeError(f"evaluator must be instance of Evaluator, got {type(rng)}")

        self.config = config

        self.statistics = statistics

        self.selection = selection
        self.variation = variation
        self.evaluator = evaluator

        self.rng = rng

        self.population = Population(
            [
                Individual(genome_generator())
                for _ in range(self.config.initial_population_size)
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
        
        #statistics
        if self.config.collect_performance_data:
            self.statistics.record(
                generation=0,
                population=self.population
            )

        pass


    def step(self, generation) -> None:
        """
        Updating one generation
        """

        #stuff is done inplace (hopefully)

        # select parents to construct offspring from
        self.selection.select(self.population)

        # create offspring
        self.variation.variate(self.population.individuals)

        # evaluate offspring
        self.evaluator.evaluate(self.population.individuals)

        self.population.clear_stats()

        #+ statistics


    def run(self):
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
        generations = self.config.n_generations

        self.initialize()

        for generation in range(generations):
            print(f"Generation {generation}/{generations}")
            self.step(generation)
            
            if self.config.collect_performance_data:
                self.statistics.record(
                    generation=generation,
                    population=self.population
                )


        best_individual = self.population.best()

        return best_individual
