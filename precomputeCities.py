import pickle
from pathlib import Path
import pandas as pd
from pympler import asizeof

from Simulation.entities import City

from paths import DATA_DIR

MATRIX_FILE = DATA_DIR / "gov_data" / "Daten Matrix Reduced.csv"
OUTPUT_FILE = DATA_DIR / "gov_data" / "cities_list_reduced_from_root_rounded.pkl"


def make_list_of_cities_from_matrix(cities):
    cities_list = []
    city_id = 0
    max = 0
    tp = 0
    for i in range(cities.shape[0]):
        for j in range(cities.shape[1]):
            value = cities.iloc[i, j]

            # skip empty cells
            if pd.isna(value):
                continue
            tp += value
            population = round(value)

            # skip cells with no population
            if population <= 0:
                continue
            if population > max:
                max = population
            city = City(
                id=city_id,
                btot=population,
                inHospital=0,
                hospitals=[]
            )
            if(city_id == 1):
                print(f"Size of a City: {asizeof.asizeof(city)}")
            cities_list.append((i, j, city))
            city_id += 1

    return cities_list, max, tp


def precompute_cities_list():
    print(f"Loading city matrix from: {MATRIX_FILE}")

    cities = pd.read_csv(MATRIX_FILE, header=None, sep=";")
    cities_list, max, tp = make_list_of_cities_from_matrix(cities)

    rows, cols = cities.shape
    total_population = sum(city.btot for _, _, city in cities_list)
    

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "wb") as file:
        pickle.dump(cities_list, file)

    print(f"Stored precomputed cities list in: {OUTPUT_FILE}")
    print(f"Matrix dimensions: {rows} x {cols}")
    print(f"Number of City objects stored: {len(cities_list)}")
    print(f"Total population stored after: {total_population}")
    print(f"Total population stored before: {tp}")
    print(f"Max: {max}")
    print(f"Size of citieslist: {asizeof.asizeof(cities_list)}")
    

    return cities_list


if __name__ == "__main__":
    precompute_cities_list()