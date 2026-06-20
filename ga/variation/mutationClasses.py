from abc import ABC, abstractmethod
import numpy as np
#from typing import Annotated
from typing import TypeAlias


from ga.individual import Individual
from ga.config import GAConfig


Gene: TypeAlias = tuple[int, int, int]
Genome: TypeAlias = list[Gene]

class MutationStrategy(ABC):
    def __init__(
            self,
            ga_config : GAConfig,
            rng : np.random.Generator
        ):
        super().__init__()
        self.height = ga_config.genome_size[0]
        self.width = ga_config.genome_size[1]
        self.n_genes = self.height * self.width
        self.rng = rng
        self.n_gene_types = ga_config.n_hospital_types + 1

    def mutate(self, individual : Individual):
        individual.fitness = None
        individual.sim_records = None
        self._mutate(individual=individual)


    @abstractmethod
    def _mutate(
        self, 
        individual : Individual,
    ) -> None:
        pass

    def get_position_from_total_index(self, idx): #total index as in the index of the city when enumerating all cities
        row = idx // self.width
        col = idx % self.width
        return row, col
    
    def get_genome_index(
            self,
            genome : Genome,
            row : int,
            col : int,
        ) -> int:
        """
        genome passed by reference
        """

        i = next((i for i, gene in enumerate(genome) if gene[1] == row and gene[2] == col), None)
        
        return i
    
    def do_point_mutation(
            self,
            i : int,
            genome : Genome,
            row : int,
            col : int,
        ):
        """
        genome passed by reference

        if i is None then the gene to mutate is not in the list -> append new hospital
        """

        n = (self.rng.integers(1, self.n_gene_types))
        

        if i is None:  #new hospital
            genome.append((n,row,col))
            return
        n = (n + genome[i][0]) % self.n_gene_types # 1 = small hospital, 2 = large hospital, 0 = no hospital
        if n == 0:  #delete hospital
            del genome[i]
            return
        
        genome[i] = (n,row,col) #change hospital
        return



class SinglePointMutation(MutationStrategy):
    def _mutate(
        self, 
        individual : Individual
    ) -> None:
        row, col = self.get_position_from_total_index(self.rng.integers(self.n_genes))
        i = self.get_genome_index(genome=individual.genome,row=row,col=col)
        self.do_point_mutation(i=i,genome=individual.genome,row=row,col=col)
        return
    


class SingePointMutationWithEqualOpportunity(MutationStrategy):
     def _mutate(
        self, 
        individual : Individual,
    ) -> None:
        if self.rng.random() < 0.5: #change / delete hospital #+ change probability of add/delete hospital
            i = self.rng.integers(len(individual.genome))
            _,r,c = individual.genome[i]
            self.do_point_mutation(i=i,genome=individual.genome,row=r,col=c)
            return
        else: # new hospital
            for k in range(5):    #!COMPUTATION, if number of hospitals stays low and the check is to heavy we can just randomly select any without check -> some small check that selected city already has hospital that might then get deleted
                row, col = self.get_position_from_total_index(self.rng.integers(self.n_genes))
                i = self.get_genome_index(genome=individual.genome,row=row,col=col)
                if i is None:
                    self.do_point_mutation(i=i,genome=individual.genome,row=row,col=col)
                    return
            print(f"No city without hospital found, no mutation done when mutation was intended. Number of hospitals: {len(individual.genome)}")



class SinglePointWanderMutation(MutationStrategy):
    def __init__(
            self,
            ga_config: GAConfig,
            rng : np.random.Generator,
            sigma : float = 6,
        ):
        super().__init__(ga_config, rng)

        assert sigma > 0, "sigma must be strictly positiv"

        self.sigma = sigma

    def _mutate(
            self, 
            individual: Individual
        ) -> None:
        """
        lets one hospital randomly wander in vicinity
        """
        i = self.rng.integers(len(individual.genome))
        t,row,col = individual.genome[i]

        for k in range(5):
            dr = int(self.rng.normal(0, self.sigma))
            dc = int(self.rng.normal(0, self.sigma))

            nr = max(0, min(self.height - 1, row + dr))
            nc = max(0, min(self.width - 1, col + dc))

            c = self.get_genome_index(genome=individual.genome,row=nr,col=nc)

            if c is None:
                individual.genome[i] = (t, nr, nc)
                return
        print(f"No city without hospital within vicinity found, no mutation done when mutation was intended. Number of hospitals: {len(individual.genome)}")


class SinglePointTeleportationMutation(MutationStrategy):
    def _mutate(
            self, 
            individual: Individual
        ) -> None:

        i = self.rng.integers(len(individual.genome))
        t,_,_ = individual.genome[i]

        for _ in range(5):
            row, col = self.get_position_from_total_index(self.rng.integers(self.n_genes))
            if self.get_genome_index(genome=individual.genome,row=row,col=col) is not None:
                continue
            individual.genome[i] = (t, row, col)
            return
        print(f"No city without hospital found, no mutation done when mutation was intended. Number of hospitals: {len(individual.genome)}")

        
class SinglePointWanderMutationWithProbabilityOfWanderMutation(MutationStrategy):
    def __init__(
            self,
            ga_config: GAConfig,
            rng : np.random.Generator,
            sigma : float = 6,
        ):
        super().__init__(ga_config, rng)

        assert sigma > 0, "sigma must be strictly positiv"

        self.sigma = sigma
        self.teleport = SinglePointTeleportationMutation(ga_config, rng)
        self.wander = SinglePointWanderMutation(ga_config, rng)

    def _mutate(
            self,
            individual : Individual,
            probability_of_normal_mutation : float = 0.5
        ) -> None:


        if self.rng.random() < probability_of_normal_mutation: #teleport mutation
            self.teleport._mutate(individual)
        else: # wandering mutation
            self.wander._mutate(individual)

    
    

class SinglePointWanderMutationWithProbabilityOfNormalMutation(MutationStrategy):

    def __init__(
            self,
            ga_config: GAConfig,
            rng : np.random.Generator,
            sigma : float = 6,
        ):
        super().__init__(ga_config, rng)

        assert sigma > 0, "sigma must be strictly positiv"

        self.sigma = sigma
        self.normal = SingePointMutationWithEqualOpportunity(ga_config, rng)
        self.wander = SinglePointWanderMutation(ga_config, rng)

    def _mutate(
            self,
            individual : Individual,
            probability_of_normal_mutation : float = 0.5
        ) -> None:

        """
        Has a chance to perform equal opportunity mutation (#! if single point mutation performs better than equal opportunity, change inheritance)
        """

        if self.rng.random() < probability_of_normal_mutation: #normal mutation
            self.normal._mutate(individual)
        else: # wandering mutation
            self.wander._mutate(individual)



            