from ga.selection.base import SelectionStrategy

from ga.individual import Individual

class TournamentSelection(SelectionStrategy):

    def __init__(self, tournament_size=3):
        self.tournament_size = tournament_size

    def select(self, population, n_parents) -> list[Individual]:
        pass