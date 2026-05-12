from ga.population import Population
from ga.individual import Individual
from ga.config import GAConfig
from ga.selection.base import SelectionStrategy
from ga.variation.base import VariationStrategy
from ga.replacement.base import ReplacementStrategy
from ga.evaluator import Evaluator
from ga.generator import Generator
from ga.analysis.statistics import GAStatistics
from ga.paths import RUNS_DIR

import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
import json




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

        self.statistics = GAStatistics()

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
        
        if self.config.collect_performance_data:
            self.statistics.record(
                generation=0,
                population=self.population.individuals
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
            
            if self.config.collect_performance_data:
                self.statistics.record(
                    generation=generation,
                    population=self.population.individuals
                )


        # save run result, config and stats (in a new folder within runs)
        dir_name = f"{self.selection.file_name()}_{self.variation.file_name()}_{self.replacement.file_name()}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{self.evaluator.file_name()}"
        dir_path = Path(RUNS_DIR) / dir_name

        # stats and plots
        if self.config.collect_performance_data:
            self.statistics.save_csv(dir_path, self.config.plot_images)

        # config
        config_dict = asdict(self.config)  #!? we should save total configuration, usually one shouldn't (but could) pass simulation config to ga, thus I think we should save the stuff in main and not here. We could use an ExperimentLogger class
        with open(dir_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=4)

        # result
        best_individual = self.population.best()
        best_dict = best_individual.to_dict()
        with open(dir_path / "config.json", "w") as f:
            json.dump(best_dict, f, indent=4)
            

        return best_individual
