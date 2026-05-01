print("Start")
import pandas as pd
from pathlib import Path
import sys


def create_bbtot_matrix(input_csv: str, output_csv: str, grid_step: int = 100) -> None:
    """
    Creates a complete coordinate matrix from STATPOP data.

    Rows are N_KOORD values.
    Columns are E_KOORD values.
    Values are BBTOT.
    Missing coordinate combinations are filled with 0.
    """

    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    # STATPOP CSV files are usually separated by semicolons
    df = pd.read_csv(input_path, sep=";")

    required_columns = ["E_KOORD", "N_KOORD", "BBTOT"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}. Available columns are: {list(df.columns)}"
        )

    df = df[required_columns]

    # Determine full coordinate range
    min_e = df["E_KOORD"].min()
    max_e = df["E_KOORD"].max()
    min_n = df["N_KOORD"].min()
    max_n = df["N_KOORD"].max()

    # Create complete coordinate axes in 100 m steps
    e_values = range(min_e, max_e + grid_step, grid_step)
    n_values = range(min_n, max_n + grid_step, grid_step)

    # Create matrix: rows = N_KOORD, columns = E_KOORD, values = BBTOT
    matrix = df.pivot_table(
        index="N_KOORD",
        columns="E_KOORD",
        values="BBTOT",
        aggfunc="sum",
        fill_value=0,
    )

    # Add missing rows and columns, then fill all holes with 0
    matrix = matrix.reindex(index=n_values, columns=e_values, fill_value=0)

    # Make sure all values are integers
    matrix = matrix.astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path)

    print("Matrix successfully created.")
    print(f"Input file: {input_csv}")
    print(f"Output file: {output_path.resolve()}")
    print(f"Matrix size: {matrix.shape[0]} rows x {matrix.shape[1]} columns")
    print(f"Total BBTOT: {matrix.to_numpy().sum()}")


if __name__ == "__main__":
    print("getMatrix.py started")
    print(f"Python executable: {sys.executable}")

    script_dir = Path(__file__).resolve().parent
    print(f"Script folder: {script_dir}")

    input_file = script_dir.parent / "STATPOP2024.csv"
    output_file = script_dir.parent / "STATPOP2024_BBTOT_matrix_100m.csv"

    print(f"Expected input file: {input_file.resolve()}")
    print(f"Expected output file: {output_file.resolve()}")

    create_bbtot_matrix(
        input_csv=str(input_file),
        output_csv=str(output_file),
        grid_step=100,
    )

    if output_file.exists():
        print(f"DONE: Output file was created here: {output_file.resolve()}")
    else:
        print("ERROR: Script finished, but output file was not created.")