from dataclasses import dataclass
import random
import pandas as pd
from entities import *
import heapq
import numpy as np
import threading
import time

@dataclass
class SimResult:
    not_admitted_count: int = 0
    not_survived_count: int = 0
    not_survived_by_day: list = None
    not_admitted_by_day: list = None
    admitted_choice_counts: dict = None
    admitted_choice_counts_by_day: list = None
    not_survived_choice_counts: dict = None
    not_survived_choice_counts_by_day: list = None

    def __post_init__(self):
        if self.not_survived_by_day is None:
            self.not_survived_by_day = []
        if self.not_admitted_by_day is None:
            self.not_admitted_by_day = []
        if self.admitted_choice_counts is None:
            self.admitted_choice_counts = {}
        if self.admitted_choice_counts_by_day is None:
            self.admitted_choice_counts_by_day = []
        if self.not_survived_choice_counts is None:
            self.not_survived_choice_counts = {}
        if self.not_survived_choice_counts_by_day is None:
            self.not_survived_choice_counts_by_day = []

class Simulation:
    def run(self):
        start_time = time.time()
        thread_id = threading.get_ident()

        print(f"Simulation started in thread {thread_id}")

        self.initi()

        while(self.steps < self.sc.END_DAYS):
            self.step()

        duration = time.time() - start_time

        print(f"Simulation finished in thread {thread_id} after {duration:.4f} seconds")

        return self.result
    def __init__(self, start_pos, cities, sc):
        self.sc = sc
        self.start_pos = start_pos
        self.cities = cities
        self.steps = 0
        self.hospitals = []
        self.rng = np.random.default_rng(self.sc.SEED)
        self.result = SimResult()
    def initi(self):
        size = 0
        for type, x, y in self.start_pos:
            self.hospitals.append(Hospital(type, size, (x, y), self.sc))
            size += 1
        self.precompute_city_hospitals()
    def step(self):
        not_survived_before = self.result.not_survived_count
        not_admitted_before = self.result.not_admitted_count
        admitted_choice_counts_before = self.result.admitted_choice_counts.copy()
        not_survived_choice_counts_before = self.result.not_survived_choice_counts.copy()

        self.update_hospitals()
        self.update_cities()

        self.result.not_survived_by_day.append(self.result.not_survived_count - not_survived_before)
        self.result.not_admitted_by_day.append(self.result.not_admitted_count - not_admitted_before)

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
    def update_hospitals(self):
        for hospital in self.hospitals:
            while (True):
                p = hospital.treat_next(self.steps)
                if(p == None):
                    break
                home_pos = p.home
                home_city = self.cities.iloc[home_pos[0], home_pos[1]]
                home_city.in_hospital -= 1
                home_city.btot += 1

    def distance(self, pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def survival_probability(self, patient, distance):
        if patient.urgency == self.sc.URGENCY_U:
            base_prob = self.sc.BASE_SURVIVAL_PROB_U
            distance_penalty = self.sc.DISTANCE_PENALTY_U
            noise_std = self.sc.SURVIVAL_NOISE_STD_U
        else:
            base_prob = self.sc.BASE_SURVIVAL_PROB_N
            distance_penalty = self.sc.DISTANCE_PENALTY_N
            noise_std = self.sc.SURVIVAL_NOISE_STD_N

        prob = base_prob - distance_penalty * distance
        prob += self.rng.normal(0, noise_std)

        return max(0.0, min(1.0, prob))

    def sorted_hospitals_by_distance(self, city_pos):
        return sorted(
            [(self.distance(city_pos, hospital.location), hospital.hos_id, hospital) for hospital in self.hospitals],
            key=lambda item: item[0]
        )

    def precompute_city_hospitals(self):
        for i in range(self.cities.shape[0]):
            for j in range(self.cities.shape[1]):
                city = self.cities.iloc[i, j]

                if city is None:
                    continue

                city.hospitals_sorted = [
                    hospital_id
                    for _, hospital_id, _ in self.sorted_hospitals_by_distance((i, j))
                ]

    def send_patient_to_nearest_available_hospital(self, patient, city):
        choice_rank = 0

        for hospital_id in city.hospitals_sorted:
            hospital = self.hospitals[hospital_id]

            if not hospital.can_treat(patient):
                continue

            choice_rank += 1

            distance_to_hospital = self.distance(patient.home, hospital.location)
            survival_probability = self.survival_probability(patient, distance_to_hospital)

            if self.rng.random() > survival_probability:
                self.result.not_survived_count += 1
                self.result.not_survived_choice_counts[choice_rank] = self.result.not_survived_choice_counts.get(choice_rank, 0) + 1
                return None

            if hospital.add_patient(patient):
                self.result.admitted_choice_counts[choice_rank] = self.result.admitted_choice_counts.get(choice_rank, 0) + 1
                return hospital.hos_id

        self.result.not_admitted_count += 1
        return None

    def update_cities(self):
        # iterate over pandas DataFrame of cities
        for i in range(self.cities.shape[0]):
            for j in range(self.cities.shape[1]):
                city = self.cities.iloc[i, j]

                # skip empty cells
                if city is None:
                    continue

                # skip cities with no available population
                if city.btot == 0:
                    continue

                # Update
                urgent_sick = self.rng.binomial(city.btot, self.sc.SICK_RATE_U)
                remaining_population = city.btot - urgent_sick
                nonurgent_sick = self.rng.binomial(remaining_population, self.sc.SICK_RATE_N)

                sick_patients = []

                for _ in range(urgent_sick):
                    days = max(1, int(self.rng.normal(self.sc.PATIENT_DAYS_U, 1)))
                    sick_patients.append(
                        Patient(
                            self.steps + days,
                            (i, j),
                            self.sc.URGENCY_U
                        )
                    )

                for _ in range(nonurgent_sick):
                    days = max(1, int(self.rng.normal(self.sc.PATIENT_DAYS_N, 2)))
                    sick_patients.append(
                        Patient(
                            self.steps + days,
                            (i, j),
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
