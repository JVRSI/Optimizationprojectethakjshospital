import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
input_file = script_dir.parent / "Daten Matrix Full only data.csv"
output_file = script_dir.parent / "statpop_heatmap.png"

matrix = pd.read_csv(
    input_file,
    sep=";",
    header=None
)

data = matrix.to_numpy()

# Mask all empty grid cells so they do not dominate the plot
data_masked = np.ma.masked_where(data == 0, data)

# Use a percentile instead of the absolute maximum.
# This makes lower-density areas visible even if a few cells have very high values.
vmax = np.percentile(data[data > 0], 99.5)

cmap = plt.cm.hot.copy()
cmap.set_bad(color="white")  # zero cells become white

plt.figure(figsize=(16, 10))
plt.imshow(
    data_masked,
    cmap=cmap,
    interpolation="nearest",
    origin="lower",
    vmin=1,
    vmax=vmax,
)

plt.colorbar(label="BBTOT", extend="max")
plt.title("STATPOP Heatmap")
plt.xlabel("E_KOORD grid index")
plt.ylabel("N_KOORD grid index")

plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.show()

print(f"Heatmap saved to: {output_file.resolve()}")
print(f"Color scale: 1 to {vmax:.0f} BBTOT")
print(f"Maximum BBTOT in data: {data.max()}")