from pathlib import Path
import argparse
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from entities import City
from simulation import Simulation
from config import SimConfig
from testHospitals import hospitalsL, hospitalsS


MATRIX_PATH = Path(
    "/Users/johannessigmund/programming/Semester 4/"
    "Optimization Project/Optimizationprojectethakjshospital/"
    "Data/gov_data/Daten Matrix Reduced.csv"
)
# /usr/bin/python3 "/Users/johannessigmund/programming/Semester 4/Optimization Project/Optimizationprojectethakjshospital/Simulation/testSim.py" --list S --seed 42 --days 20 --plot

def load_population_matrix(matrix_path: Path) -> np.ndarray:
    """
    Loads the reduced population matrix as numeric data.

    Non-numeric or empty cells are converted to 0.
    """
    df = pd.read_csv(matrix_path, header=None, sep=None, engine="python")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)

    matrix = df.values.astype(float)
    return matrix


def matrix_to_city_dataframe(matrix: np.ndarray) -> pd.DataFrame:
    """
    Converts the numeric population matrix into a pandas DataFrame of City objects.

    Empty cells / cells with population <= 0 are stored as None.
    Non-empty cells are stored as City objects.
    """
    rows, cols = matrix.shape

    city_grid = []

    city_id = 0

    for row in range(rows):
        city_row = []

        for col in range(cols):
            population = int(matrix[row, col])

            if population <= 0:
                city_row.append(None)
            else:
                city_row.append(
                    City(
                        id=city_id,
                        btot=population,
                        inHospital=0,
                        hospitals=[]
                    )
                )
                city_id += 1

        city_grid.append(city_row)

    return pd.DataFrame(city_grid)


def convert_hospitals_xy_to_row_col(hospitals_xy: np.ndarray) -> np.ndarray:
    """
    testHospitals.py stores hospitals as:
        [size, x, y]

    Simulation uses matrix positions as:
        (row, col) = (y, x)

    Therefore we convert:
        [size, x, y] -> [size, y, x]
    """
    converted = []

    for size, x, y in hospitals_xy:
        converted.append([int(size), int(y), int(x)])

    return np.array(converted, dtype=int)


def validate_hospitals_inside_matrix(hospitals_row_col: np.ndarray, matrix: np.ndarray) -> None:
    rows, cols = matrix.shape

    outside = []

    for idx, hospital in enumerate(hospitals_row_col):
        size, row, col = hospital

        if not (0 <= row < rows and 0 <= col < cols):
            outside.append((idx, size, row, col))

    if outside:
        print("WARNING: Some hospitals are outside the matrix:")
        for item in outside:
            print(item)
    else:
        print("All hospitals are inside the matrix.")


def plot_results(result, title: str) -> None:
    days = np.arange(len(result.not_admitted_by_day))

    plt.figure(figsize=(12, 6))
    plt.plot(days, result.not_admitted_by_day, label="Not admitted per day")
    plt.plot(days, result.not_survived_by_day, label="Not survived per day")

    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("Number of patients")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# New plotting functions for admitted choices
def plot_admitted_choices(result, title: str) -> None:
    """
    Plots admitted patients and patients who did not survive side-by-side
    for each hospital choice rank.
    """
    admitted_counts = getattr(result, "admitted_choice_counts", {})
    died_counts = getattr(result, "not_survived_choice_counts", {})

    if not admitted_counts and not died_counts:
        print("No admitted/not-survived choice-count data available to plot.")
        return

    choices = sorted(set(admitted_counts.keys()) | set(died_counts.keys()))
    admitted_values = [admitted_counts.get(choice, 0) for choice in choices]
    died_values = [died_counts.get(choice, 0) for choice in choices]

    x_positions = np.arange(len(choices))
    bar_width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar(x_positions - bar_width / 2, admitted_values, width=bar_width, label="Admitted")
    plt.bar(x_positions + bar_width / 2, died_values, width=bar_width, label="Did not survive")

    plt.title(title)
    plt.xlabel("Hospital choice rank")
    plt.ylabel("Number of patients")
    plt.xticks(x_positions, choices)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_admitted_choices_by_day(result, title: str) -> None:
    """
    Plots daily admitted patients by hospital choice rank.
    """
    if not hasattr(result, "admitted_choice_counts_by_day") or not result.admitted_choice_counts_by_day:
        print("No admitted choice-count-by-day data available to plot.")
        return

    all_choices = sorted(
        {
            choice
            for day_counts in result.admitted_choice_counts_by_day
            for choice in day_counts.keys()
        }
    )

    if not all_choices:
        print("No admitted choice-count-by-day data available to plot.")
        return

    days = np.arange(len(result.admitted_choice_counts_by_day))

    plt.figure(figsize=(12, 6))

    for choice in all_choices:
        values = [day_counts.get(choice, 0) for day_counts in result.admitted_choice_counts_by_day]
        plt.plot(days, values, marker="o", label=f"Choice {choice}")

    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("Number of admitted patients")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_not_survived_choices_by_day(result, title: str) -> None:
    """
    Plots daily not-survived patients by attempted hospital choice rank.
    """
    if not hasattr(result, "not_survived_choice_counts_by_day") or not result.not_survived_choice_counts_by_day:
        print("No not-survived choice-count-by-day data available to plot.")
        return

    all_choices = sorted(
        {
            choice
            for day_counts in result.not_survived_choice_counts_by_day
            for choice in day_counts.keys()
        }
    )

    if not all_choices:
        print("No not-survived choice-count-by-day data available to plot.")
        return

    days = np.arange(len(result.not_survived_choice_counts_by_day))

    plt.figure(figsize=(12, 6))

    for choice in all_choices:
        values = [day_counts.get(choice, 0) for day_counts in result.not_survived_choice_counts_by_day]
        plt.plot(days, values, marker="o", label=f"Choice {choice}")

    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("Number of patients who did not survive")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_hospitals_on_matrix(matrix: np.ndarray, hospitals_xy: np.ndarray, title: str) -> None:
    sizes = hospitals_xy[:, 0]
    x = hospitals_xy[:, 1]
    y = hospitals_xy[:, 2]

    dot_sizes = np.where(sizes == 2, 90, 45)

    plt.figure(figsize=(13, 8))

    plt.imshow(
        matrix,
        origin="lower",
        cmap="Greys",
        interpolation="none"
    )

    plt.scatter(
        x,
        y,
        s=dot_sizes,
        c="red",
        edgecolors="black",
        linewidths=0.8,
        label="Hospitals"
    )

    major = sizes == 2

    plt.scatter(
        x[major],
        y[major],
        s=dot_sizes[major] + 50,
        facecolors="none",
        edgecolors="yellow",
        linewidths=1.5,
        label="Large hospitals"
    )

    plt.title(title)
    plt.xlabel("Reduced matrix x-coordinate / column")
    plt.ylabel("Reduced matrix y-coordinate / row")

    plt.xlim(0, matrix.shape[1])
    plt.ylim(0, matrix.shape[0])

    plt.grid(color="lightblue", linewidth=0.3, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_simulation(hospital_list_name: str, seed: Optional[int], days: Optional[int], plot: bool) -> None:
    if hospital_list_name.upper() == "L":
        hospitals_xy = hospitalsL
        label = "hospitalsL"
    elif hospital_list_name.upper() == "S":
        hospitals_xy = hospitalsS
        label = "hospitalsS"
    else:
        raise ValueError("hospital_list_name must be either 'L' or 'S'.")

    print(f"Selected hospital list: {label}")
    print(f"Number of hospitals: {len(hospitals_xy)}")

    matrix = load_population_matrix(MATRIX_PATH)

    print("Matrix shape:", matrix.shape)
    print("Matrix dtype:", matrix.dtype)
    print("Matrix min/max:", np.min(matrix), np.max(matrix))
    print("Total population in matrix:", int(np.sum(matrix)))

    cities = matrix_to_city_dataframe(matrix)

    hospitals_row_col = convert_hospitals_xy_to_row_col(hospitals_xy)

    validate_hospitals_inside_matrix(hospitals_row_col, matrix)

    sc = SimConfig()

    if seed is not None:
        sc.SEED = seed

    if days is not None:
        sc.END_DAYS = days

    simulation = Simulation(
        start_pos=hospitals_row_col,
        cities=cities,
        sc=sc
    )

    result = simulation.run()

    print()
    print("Simulation finished")
    print("-------------------")
    print("Hospital list:", label)
    print("Days simulated:", sc.END_DAYS)
    print("Total not admitted:", result.not_admitted_count)
    print("Total not survived:", result.not_survived_count)
    print("Not admitted by day:", result.not_admitted_by_day)
    print("Not survived by day:", result.not_survived_by_day)

    if hasattr(result, "admitted_choice_counts"):
        print("Admitted choice counts:", dict(sorted(result.admitted_choice_counts.items())))
    else:
        print("Admitted choice counts: not available")

    if hasattr(result, "admitted_choice_counts_by_day"):
        print("Admitted choice counts by day:", result.admitted_choice_counts_by_day)
    else:
        print("Admitted choice counts by day: not available")

    if hasattr(result, "not_survived_choice_counts"):
        print("Not survived choice counts:", dict(sorted(result.not_survived_choice_counts.items())))
    else:
        print("Not survived choice counts: not available")

    if hasattr(result, "not_survived_choice_counts_by_day"):
        print("Not survived choice counts by day:", result.not_survived_choice_counts_by_day)
    else:
        print("Not survived choice counts by day: not available")

    if plot:
        plot_hospitals_on_matrix(
            matrix,
            hospitals_xy,
            title=f"{label} on reduced matrix"
        )

        plot_results(
            result,
            title=f"Simulation results for {label}"
        )

        plot_admitted_choices(
            result,
            title=f"Admitted vs not-survived patients by hospital choice rank for {label}"
        )

        plot_admitted_choices_by_day(
            result,
            title=f"Daily admitted patients by hospital choice rank for {label}"
        )

        plot_not_survived_choices_by_day(
            result,
            title=f"Daily not-survived patients by attempted hospital choice rank for {label}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--list",
        choices=["L", "S"],
        default="S",
        help="Choose hospital list: L for hospitalsL, S for hospitalsS."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed."
    )

    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Optional number of simulation days. If not set, uses SimConfig.END_DAYS."
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show plots after the simulation."
    )

    args = parser.parse_args()

    run_simulation(
        hospital_list_name=args.list,
        seed=args.seed,
        days=args.days,
        plot=args.plot
    )