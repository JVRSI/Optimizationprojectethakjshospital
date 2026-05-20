import pickle
from pathlib import Path
import pandas as pd

from Simulation.entities import City

from paths import DATA_DIR

MATRIX_FILE = DATA_DIR / "gov_data" / "Daten Matrix Reduced.csv"
OUTPUT_FILE = DATA_DIR / "gov_data" / "cities_list_reduced_from_root.pkl"


def make_list_of_cities_from_matrix(cities):
    cities_list = []
    city_id = 0

    for i in range(cities.shape[0]):
        for j in range(cities.shape[1]):
            value = cities.iloc[i, j]

            # skip empty cells
            if pd.isna(value):
                continue

            population = value

            # skip cells with no population
            if population <= 0:
                continue

            city = City(
                id=city_id,
                btot=population,
                inHospital=0,
                hospitals=[]
            )

            cities_list.append((i, j, city))
            city_id += 1

    return cities_list


def precompute_cities_list():
    print(f"Loading city matrix from: {MATRIX_FILE}")

    cities = pd.read_csv(MATRIX_FILE, header=None, sep=";")
    cities_list = make_list_of_cities_from_matrix(cities)

    rows, cols = cities.shape
    total_population = sum(city.btot for _, _, city in cities_list)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "wb") as file:
        pickle.dump(cities_list, file)

    print(f"Stored precomputed cities list in: {OUTPUT_FILE}")
    print(f"Matrix dimensions: {rows} x {cols}")
    print(f"Number of City objects stored: {len(cities_list)}")
    print(f"Total population stored: {total_population}")

    return cities_list


if __name__ == "__main__":
    precompute_cities_list()