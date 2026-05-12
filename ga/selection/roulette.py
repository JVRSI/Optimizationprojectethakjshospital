from ga.selection.base import SelectionStrategy

from ga.individual import Individual


class RouletteSelection(SelectionStrategy):

    def select(self, population, n_parents) -> list[Individual]:
        pass