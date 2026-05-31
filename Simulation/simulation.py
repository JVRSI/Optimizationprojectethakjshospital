from dataclasses import dataclass, fields
from Simulation.entities import *
from Simulation import SimConfig
try:
    from Simulation.entities import *
    from Simulation import SimConfig
except ImportError:
    from entities import *
    from config import SimConfig
import numpy as np
import threading
import os
import time

class Simulation:
    # ---------- Initialization -----------------------------------------------------------------------------------
    def __init__(self, start_pos, sc:SimConfig, cities_list=None, cities=None, rng = None):
        self.sc = sc
        self.start_pos = start_pos

        if cities_list is None:
            cities_list = []

        if cities is None and len(cities_list) == 0:
            raise ValueError("Either cities or cities_list must be provided.")

        self.cities = cities
        self.cities_list = cities_list
        self.steps = 0
        self.hospitals = []
        if(rng is None): 
            self.rng = np.random.default_rng(self.sc.SEED)
        else:
            self.rng = rng
        self.result = SimResult()

        self.step_times = []
        self.update_hospitals_times = []
        self.update_cities_times = []


        #for analysis
        self.do_analysis = True
        self.survival_probability_per_distance:list[(int,float,float)] = []
        self.survival_result_by_probability:list[(int,float,bool)] = []

    def initi(self):
        """
        Create hospital objects from the provided start positions and
        precompute nearest-hospital rankings for all cities.
        """
        size = 0
        for hospital_type, x, y in self.start_pos:
            if(hospital_type > 2 or hospital_type < 1):
                continue
            self.hospitals.append(Hospital(hospital_type, size, (x, y), self.sc))
            size += 1
        self.precompute_city_hospitals()

    # ---------- Start of Sim and Step -----------------------------------------------------------------------------------
    def run(self, log=True):
        """
        Execute the full simulation over all configured days
        and return the final fitness score.
        log = True -> prints Timing and other things
        Fitness score lower better
        returns 1.0 if no hospitals
        """
        start_time = time.time()
        thread_id = threading.get_ident()

        if log:
            #start_time = time.time()
            #thread_id = threading.get_ident()
            process_id = os.getpid()
            print(f"Simulation started in thread {thread_id}, process {process_id}")

        self.initi()
        if len(self.cities_list) == 0:
            self.make_list_of_cities()

        # No hospital fitness = 1
        if(len(self.hospitals) == 0):
            return 1.0
        
        # Run simulation
        while(self.steps < self.sc.END_DAYS):
            self.step()

        duration = time.time() - start_time
        if log:
            print(f"Simulation finished in thread {thread_id}, in process {process_id}, after {duration:.4f} seconds")
            self.print_timing_results()
        
        self.duration = duration

        return self.calculate_fitness()
    
    def step(self):
        """
        Execute one simulation day:
        1. discharge completed patients
        2. generate new sick patients
        3. attempt hospital admissions
        4. record daily statistics
        """
        step_start_time = time.perf_counter()
        not_survived_before = self.result.not_survived_count
        not_admitted_before = self.result.not_admitted_count
        admitted_before = self.result.admitted_count
        admitted_choice_counts_before = self.result.admitted_choice_counts.copy()
        not_survived_choice_counts_before = self.result.not_survived_choice_counts.copy()

        update_hospitals_start_time = time.perf_counter()
        self.update_hospitals()
        self.update_hospitals_times.append(time.perf_counter() - update_hospitals_start_time)

        update_cities_start_time = time.perf_counter()
        self.update_cities()
        self.update_cities_times.append(time.perf_counter() - update_cities_start_time)

        self.result.not_survived_by_day.append(self.result.not_survived_count - not_survived_before)
        self.result.not_admitted_by_day.append(self.result.not_admitted_count - not_admitted_before)
        self.result.admitted_by_day.append(self.result.admitted_count - admitted_before)

        admitted_choice_counts_today = {}
        for choice_rank, count_after in self.result.admitted_choice_counts.items():
            count_before = admitted_choice_counts_before.get(choice_rank, 0)
            admitted_choice_counts_today[choice_rank] = count_after - count_before

        self.result.admitted_choice_counts_by_day.append(admitted_choice_counts_today)

        not_survived_choice_counts_today = {}
        for choice_rank, count_after in self.result.not_survived_choice_counts.items():
            count_before = not_survived_choice_counts_before.get(choice_rank, 0)
            not_survived_choice_counts_today[choice_rank] = count_after - count_before

        self.result.not_survived_choice_counts_by_day.append(not_survived_choice_counts_today)

        self.steps += 1
        self.step_times.append(time.perf_counter() - step_start_time)
    
    # ---------- Fitness -----------------------------------------------------------------------------------
    def calculate_fitness(self):
        """
        Combine mortality, admission failure, travel distance,
        hospital usage, and cost into a normalized fitness score.

        Lower fitness values are better.
        """
        total_patients = self.result.admitted_count + self.result.not_survived_count + self.result.not_admitted_count

        if total_patients == 0:
            return 0.0

        average_admitted_distance = 0.0
        if len(self.result.admitted_travel_distances) > 0:
            average_admitted_distance = sum(self.result.admitted_travel_distances) / len(self.result.admitted_travel_distances)

        average_not_survived_distance = 0.0
        if len(self.result.not_survived_travel_distances) > 0:
            average_not_survived_distance = sum(self.result.not_survived_travel_distances) / len(self.result.not_survived_travel_distances)

        total_admitted_choice_rank = sum(
            choice_rank * count
            for choice_rank, count in self.result.admitted_choice_counts.items()
        )
        average_admitted_choice_rank = 0.0
        if self.result.admitted_count > 0:
            average_admitted_choice_rank = total_admitted_choice_rank / self.result.admitted_count

        total_not_survived_choice_rank = sum(
            choice_rank * count
            for choice_rank, count in self.result.not_survived_choice_counts.items()
        )
        average_not_survived_choice_rank = 0.0
        if self.result.not_survived_count > 0:
            average_not_survived_choice_rank = total_not_survived_choice_rank / self.result.not_survived_count

        used_hospitals = len(self.result.admitted_per_hospital)
        total_hospitals = len(self.hospitals)
        unused_hospitals = total_hospitals - used_hospitals

        total_hospital_cost = sum(hospital.cost for hospital in self.hospitals)

        death_rate = self.result.not_survived_count / total_patients
        not_admitted_rate = self.result.not_admitted_count / total_patients
        urgent_death_rate = self.result.not_survived_urgent / total_patients
        urgent_not_admitted_rate = self.result.not_admitted_urgent / total_patients

        max_distance = 408.62

        normalized_admitted_distance = average_admitted_distance / max_distance
        normalized_not_survived_distance = average_not_survived_distance / max_distance

        max_choice_rank = max(1, len(self.hospitals))

        normalized_admitted_choice_rank = average_admitted_choice_rank / max_choice_rank
        normalized_not_survived_choice_rank = average_not_survived_choice_rank / max_choice_rank


        normalized_cost = 0.0
        
        normalized_cost = total_hospital_cost / self.sc.TOTALCOST 

        normalized_unused_hospitals = 0.0
        if total_hospitals > 0:
            normalized_unused_hospitals = unused_hospitals / total_hospitals

        fitness = 0.0

        self.result.death_rate = death_rate
        self.result.not_admitted_rate = not_admitted_rate
        self.result.urgent_death_rate = urgent_death_rate
        self.result.urgent_not_admitted_rate = urgent_not_admitted_rate
        self.result.normalized_admitted_distance = normalized_admitted_distance
        self.result.normalized_not_survived_distance = normalized_not_survived_distance
        self.result.normalized_admitted_choice_rank = normalized_admitted_choice_rank
        self.result.normalized_not_survived_choice_rank = normalized_not_survived_choice_rank
        self.result.normalized_unused_hospitals = normalized_unused_hospitals
        self.result.normalized_cost = normalized_cost

        fitness += self.sc.death_rate_factor * death_rate
        fitness += self.sc.not_admitted_rate_factor * not_admitted_rate
        fitness += self.sc.urgent_death_rate_factor * urgent_death_rate
        fitness += self.sc.urgent_not_admitted_rate_factor * urgent_not_admitted_rate
        fitness += self.sc.normalized_admitted_distance_factor * normalized_admitted_distance
        fitness += self.sc.normalized_not_survived_distance_factor * normalized_not_survived_distance
        fitness += self.sc.normalized_admitted_choice_rank_factor * normalized_admitted_choice_rank
        fitness += self.sc.normalized_not_survived_choice_rank_factor * normalized_not_survived_choice_rank
        fitness += self.sc.normalized_unused_hospitals_factor * normalized_unused_hospitals
        fitness += self.sc.normalized_cost_factor * normalized_cost

        return fitness

    # ---------- Main Logic -----------------------------------------------------------------------------------
    def update_hospitals(self):
        for hospital in self.hospitals:
            while (True):
                p = hospital.treat_next(self.steps)
                if(p == None):
                    break
                home_city = p.home
                home_city.in_hospital -= 1
                home_city.btot += 1

    def update_cities(self):
        """
        Generate new sick patients for each city and attempt
        to admit them to hospitals.
        """
        for city in self.cities_list:

            # skip empty cells or with no available population
            if city is None or city.btot == 0:
                continue

            # Update
            urgent_sick = self.rng.binomial(city.btot, self.sc.SICK_RATE_U)
            remaining_population = city.btot - urgent_sick
            nonurgent_sick = self.rng.binomial(remaining_population, self.sc.SICK_RATE_N)

            sick_patients:list[Patient] = []

            for _ in range(urgent_sick):
                days = max(1, int(self.rng.normal(self.sc.PATIENT_DAYS_U, 1)))
                sick_patients.append(
                    Patient(
                        self.steps + days,
                        city,
                        self.sc.URGENCY_U
                    )
                )

            for _ in range(nonurgent_sick):
                days = max(1, int(self.rng.normal(self.sc.PATIENT_DAYS_N, 2)))
                sick_patients.append(
                    Patient(
                        self.steps + days,
                        city,
                        self.sc.URGENCY_N
                    )
                )

            for patient in sick_patients:
                # patient temporarily leaves the city population
                city.btot -= 1

                hospital_id = self.send_patient_to_nearest_available_hospital(patient, city)

                if hospital_id is not None:
                    city.in_hospital += 1
                else:
                    # if the patient is not admitted or does not survive the trip,
                    # return them directly to the city population
                    city.btot += 1

    # ---------- Helper -----------------------------------------------------------------------------------
    def print_timing_results(self):
        print("\nTiming results per iteration:")

        mean_step_time = sum(self.step_times) / len(self.step_times) if len(self.step_times) > 0 else 0.0
        mean_update_hospitals_time = sum(self.update_hospitals_times) / len(self.update_hospitals_times) if len(self.update_hospitals_times) > 0 else 0.0
        mean_update_cities_time = sum(self.update_cities_times) / len(self.update_cities_times) if len(self.update_cities_times) > 0 else 0.0

        print("\nMean timing results:")
        print(f"Mean step time: {mean_step_time:.6f} seconds")
        print(f"Mean update_hospitals time: {mean_update_hospitals_time:.6f} seconds")
        print(f"Mean update_cities time: {mean_update_cities_time:.6f} seconds")

    def get_result(self):
        return self.result

    def distance(self, pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    def distance_squared(self, pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) 

    def survival_probability(self, patient : Patient, distance):
        if patient.urgency == self.sc.URGENCY_U:
            base_prob = self.sc.BASE_SURVIVAL_PROB_U
            distance_penalty = self.sc.DISTANCE_PENALTY_U
            noise_std = self.sc.SURVIVAL_NOISE_STD_U
            #prob = base_prob - (distance_penalty * distance)
            prob = base_prob * np.e ** (-distance_penalty*distance)
        else:
            base_prob = self.sc.BASE_SURVIVAL_PROB_N
            distance_penalty = self.sc.DISTANCE_PENALTY_N
            noise_std = self.sc.SURVIVAL_NOISE_STD_N
            prob = base_prob - (distance_penalty * distance)

        prob += self.rng.normal(0, noise_std)

        if self.do_analysis:
            self.survival_probability_per_distance.append((patient.urgency, distance**0.5, prob))

        return max(0.0, min(1.0, prob))

    def record_admission(self, patient, hospital, choice_rank, distance_to_hospital):
        self.result.admitted_count += 1
        self.result.total_travel_distance += distance_to_hospital
        self.result.admitted_travel_distances.append(distance_to_hospital)
        self.result.admitted_choice_counts[choice_rank] = self.result.admitted_choice_counts.get(choice_rank, 0) + 1
        self.result.admitted_per_hospital[hospital.hos_id] = self.result.admitted_per_hospital.get(hospital.hos_id, 0) + 1

        if patient.urgency == self.sc.URGENCY_U:
            self.result.admitted_urgent += 1
        else:
            self.result.admitted_nonurgent += 1

    def record_not_survived(self, patient, hospital, choice_rank, distance_to_hospital):
        self.result.not_survived_count += 1
        self.result.total_travel_distance += distance_to_hospital
        self.result.not_survived_travel_distances.append(distance_to_hospital)
        self.result.not_survived_choice_counts[choice_rank] = self.result.not_survived_choice_counts.get(choice_rank, 0) + 1
        self.result.rejected_per_hospital[hospital.hos_id] = self.result.rejected_per_hospital.get(hospital.hos_id, 0) + 1

        if patient.urgency == self.sc.URGENCY_U:
            self.result.not_survived_urgent += 1
        else:
            self.result.not_survived_nonurgent += 1

    def record_not_admitted(self, patient):
        self.result.not_admitted_count += 1

        if patient.urgency == self.sc.URGENCY_U:
            self.result.not_admitted_urgent += 1
        else:
            self.result.not_admitted_nonurgent += 1

    def sorted_hospitals_by_distance(self, city_pos):
        values = [
            self.distance_squared(city_pos, hospital.location)
            for hospital in self.hospitals
        ]
        return sorted(range(len(values)), key=values.__getitem__)
    
    #is faster, only returns hospital id
        #return sorted(
        #    [(self.distance_squared(city_pos, hospital.location), hospital.hos_id, hospital) for hospital in self.hospitals],
        #    key=lambda item: item[0]
        #)

    def precompute_city_hospitals(self):
        if self.cities_list:
            for city in self.cities_list:
                if city is None:
                    continue

                city.hospitals_sorted = [
                    hospital_id
                    for hospital_id in self.sorted_hospitals_by_distance(city.location)
                ]
            return

        for i in range(self.cities.shape[0]):
            for j in range(self.cities.shape[1]):
                city = self.cities.iloc[i, j]

                if city is None:
                    continue

                city.hospitals_sorted = [
                    hospital_id
                    for hospital_id in self.sorted_hospitals_by_distance((i, j))
                ]

    def send_patient_to_nearest_available_hospital(self, patient:Patient, city):
        """
        Try hospitals in ascending distance order until:
        - patient dies during travel -> deathrate
        - patient is admitted
        - no valid hospital remains -> not admittedrate
        """
        choice_rank = 0

        for hospital_id in city.hospitals_sorted:
            hospital = self.hospitals[hospital_id]

            if not hospital.can_treat(patient):
                continue


            choice_rank += 1

            distance_to_hospital = self.distance_squared(patient.home.location, hospital.location)
            survival_probability = self.survival_probability(patient, distance_to_hospital)


            rn = self.rng.random()
            if self.do_analysis:
                self.survival_result_by_probability.append((patient.urgency,survival_probability,(rn<=survival_probability)))

            if rn > survival_probability:
                self.record_not_survived(patient, hospital, choice_rank, distance_to_hospital)
                return None

            if hospital.add_patient(patient):
                self.record_admission(patient, hospital, choice_rank, distance_to_hospital)
                return hospital.hos_id

            print(patient.urgency)
            self.result.rejected_per_hospital[hospital.hos_id] = self.result.rejected_per_hospital.get(hospital.hos_id, 0) + 1

        self.record_not_admitted(patient)
        return None
    
    def make_list_of_cities(self):
        self.cities_list = []
        for i in range(self.cities.shape[0]):
            for j in range(self.cities.shape[1]):
                city = self.cities.iloc[i, j]

                # skip empty cells
                if city is None:
                    continue

                # skip cities with no available population
                if city.btot == 0:
                    continue
                self.cities_list.append(city)


# Result Class
@dataclass
class SimResultScalar:
    not_admitted_count: int = 0
    not_survived_count: int = 0
    admitted_count: int = 0

    not_survived_urgent: int = 0
    not_survived_nonurgent: int = 0
    not_admitted_urgent: int = 0
    not_admitted_nonurgent: int = 0
    admitted_urgent: int = 0
    admitted_nonurgent: int = 0

    total_travel_distance: float = 0.0

    death_rate: float = None
    not_admitted_rate: float = None
    urgent_death_rate: float = None
    urgent_not_admitted_rate: float = None
    normalized_admitted_distance: float = None
    normalized_not_survived_distance: float = None
    normalized_admitted_choice_rank: float = None
    normalized_not_survived_choice_rank: float = None
    normalized_unused_hospitals: float = None
    normalized_cost: float = None

    def __str__(self):
        RED        = "\033[38;5;160m"
        GREEN      = "\033[38;5;10m"
        YELLOW     = "\033[33m"
        BLUE       = "\033[34m"
        MAGENTA    = "\033[38;5;198m"
        CYAN       = "\033[36m"
        RESET      = "\033[0m"
        BOLD_BLACK = "\033[30;1m"
        return(
            f"{BOLD_BLACK}Not Normalized{RESET}\n"
            f"{GREEN  }            Admitted total:{RESET} {self.admitted_count:7d}\n"
            f"{GREEN  }           Admitted urgent:{RESET} {self.admitted_urgent:7d}\n"
            f"{GREEN  }       Admitted non-urgent:{RESET} {self.admitted_nonurgent:7d}\n"
            f"{RED    }        Not admitted total:{RESET} {self.not_admitted_count:7d}\n"
            f"{RED    }       Not admitted urgent:{RESET} {self.not_admitted_urgent:7d}\n"
            f"{RED    }   Not admitted non-urgent:{RESET} {self.not_admitted_nonurgent:7d}\n"
            f"{MAGENTA}        Not survived total:{RESET} {self.not_survived_count:7d}\n"
            f"{MAGENTA}       Not survived urgent:{RESET} {self.not_survived_urgent:7d}\n"
            f"{MAGENTA}   Not survived non-urgent:{RESET} {self.not_survived_nonurgent:7d}\n"
            f"{CYAN   }     Travel distance total:{RESET} {self.total_travel_distance:10.2f}\n"
            f"{BOLD_BLACK}Normalized{RESET}\n"
            f"{RED    }        Not admitted total:{RESET} {self.not_admitted_rate:9.5f}\n"
            f"{RED    }       Not admitted urgent:{RESET} {self.urgent_not_admitted_rate:9.5f}\n"
            f"{MAGENTA}        Not survived total:{RESET} {self.death_rate:9.5f}\n"
            f"{MAGENTA}       Not survived urgent:{RESET} {self.urgent_death_rate:9.5f}\n"
            f"{CYAN   }         Admitted distance:{RESET} {self.normalized_admitted_distance:9.5f}\n"
            f"{CYAN   }     Not survived distance:{RESET} {self.normalized_not_survived_distance:9.5f}\n"
            f"{CYAN   }       Not survived choice:{RESET} {self.normalized_not_survived_choice_rank:9.5f}\n"
            f"{CYAN   }          Unused Hospitals:{RESET} {self.normalized_unused_hospitals:9.5f}\n"
            f"{CYAN   }                      Cost:{RESET} {self.normalized_cost:9.5f}\n"
        
        )


@dataclass
class SimResult(SimResultScalar):
    admitted_travel_distances: list = None
    not_survived_travel_distances: list = None

    not_survived_by_day: list = None
    not_admitted_by_day: list = None
    admitted_by_day: list = None

    admitted_choice_counts: dict = None
    admitted_choice_counts_by_day: list = None
    not_survived_choice_counts: dict = None
    not_survived_choice_counts_by_day: list = None

    admitted_per_hospital: dict = None
    rejected_per_hospital: dict = None

    def __post_init__(self):
        if self.admitted_travel_distances is None:
            self.admitted_travel_distances = []
        if self.not_survived_travel_distances is None:
            self.not_survived_travel_distances = []
        if self.not_survived_by_day is None:
            self.not_survived_by_day = []
        if self.not_admitted_by_day is None:
            self.not_admitted_by_day = []
        if self.admitted_by_day is None:
            self.admitted_by_day = []
        if self.admitted_choice_counts is None:
            self.admitted_choice_counts = {}
        if self.admitted_choice_counts_by_day is None:
            self.admitted_choice_counts_by_day = []
        if self.not_survived_choice_counts is None:
            self.not_survived_choice_counts = {}
        if self.not_survived_choice_counts_by_day is None:
            self.not_survived_choice_counts_by_day = []
        if self.admitted_per_hospital is None:
            self.admitted_per_hospital = {}
        if self.rejected_per_hospital is None:
            self.rejected_per_hospital = {}

    def to_scalar(self) -> SimResultScalar:
        return SimResultScalar(
            **{
                f.name: getattr(self, f.name)
                for f in fields(SimResultScalar)
            }
        )
