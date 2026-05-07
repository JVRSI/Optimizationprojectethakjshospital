from dataclasses import dataclass
import random
import pandas as pd
from entities import Patient, City, Hospital
from config import *
import heapq
import numpy as np

class Simulation:
    def run(self):
        print("a Simulation has started")
        self.initi()
        while(self.steps < END_DAYS):
            self.step()
        print("a Simulation has finished")
    def __init__(self, start_pos, cities):
        self.start_pos = start_pos
        self.cities = cities
        self.steps = 0
        self.hospitals = []
        self.rng = np.random.default_rng(SEED)
        self.not_admitted_count = 0
        self.not_survived_count = 0
        self.not_survived_by_day = []
        self.not_admitted_by_day = []
    def initi(self):
        size = 0
        for type, (x, y) in self.start_pos:
            self.hospitals.append(Hospital(type, size, (x, y)))
            size += 1
        self.precompute_city_hospitals()
    def step(self):
        not_survived_before = self.not_survived_count
        not_admitted_before = self.not_admitted_count

        self.update_hospitals()
        self.update_cities()

        self.not_survived_by_day.append(self.not_survived_count - not_survived_before)
        self.not_admitted_by_day.append(self.not_admitted_count - not_admitted_before)

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
        if patient.urgency == URGENCY_U:
            base_prob = BASE_SURVIVAL_PROB_U
            distance_penalty = DISTANCE_PENALTY_U
            noise_std = SURVIVAL_NOISE_STD_U
        else:
            base_prob = BASE_SURVIVAL_PROB_N
            distance_penalty = DISTANCE_PENALTY_N
            noise_std = SURVIVAL_NOISE_STD_N

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
        for hospital_id in city.hospitals_sorted:
            hospital = self.hospitals[hospital_id]

            if not hospital.can_treat(patient):
                continue

            distance_to_hospital = self.distance(patient.home, hospital.location)
            survival_probability = self.survival_probability(patient, distance_to_hospital)

            if self.rng.random() > survival_probability:
                self.not_survived_count += 1
                return None

            if hospital.add_patient(patient):
                return hospital.hos_id

        self.not_admitted_count += 1
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
                urgent_sick = self.rng.binomial(city.btot, SICK_RATE_U)
                remaining_population = city.btot - urgent_sick
                nonurgent_sick = self.rng.binomial(remaining_population, SICK_RATE_N)

                sick_patients = []

                for _ in range(urgent_sick):
                    days = max(1, int(self.rng.normal(PATIENT_DAYS_U, 1)))
                    sick_patients.append(
                        Patient(
                            self.steps + days,
                            (i, j),
                            URGENCY_U
                        )
                    )

                for _ in range(nonurgent_sick):
                    days = max(1, int(self.rng.normal(PATIENT_DAYS_N, 2)))
                    sick_patients.append(
                        Patient(
                            self.steps + days,
                            (i, j),
                            URGENCY_N
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
