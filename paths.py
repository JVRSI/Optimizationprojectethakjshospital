from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]

RUNS_DIR = PROJECT_ROOT / "runs"
DATA_DIR = PROJECT_ROOT / "Data"
MATRIX_PATH = PROJECT_ROOT  / DATA_DIR / "gov_data/Daten Matrix Reduced.csv"
LOCAL_RUN = PROJECT_ROOT / "local_runs"