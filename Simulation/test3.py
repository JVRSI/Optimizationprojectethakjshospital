import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

matrix_path = Path(
    "/Users/johannessigmund/programming/Semester 4/"
    "Optimization Project/Optimizationprojectethakjshospital/"
    "Data/gov_data/Daten Matrix Reduced.csv"
)

# Load reduced matrix robustly
df = pd.read_csv(matrix_path, header=None, sep=None, engine="python")

# Convert all cells to numeric values
df = df.apply(pd.to_numeric, errors="coerce")

# Replace non-numeric / empty cells with 0
df = df.fillna(0)

matrix = df.values.astype(float)

print("Matrix shape:", matrix.shape)
print("Matrix dtype:", matrix.dtype)
print("Min / max:", np.nanmin(matrix), np.nanmax(matrix))

hospitals2 = np.array([
    [2, 197, 172],
    [2, 192, 170],
    [2, 210, 189],
    [1, 205, 173],
    [1, 216, 167],
    [1, 195, 153],
    [2, 125, 192],
    [1, 126, 193],
    [1, 128, 189],
    [2, 114, 123],
    [1, 115, 126],
    [1, 126, 103],
    [1, 149, 93],
    [1, 99, 146],
    [1, 131, 137],
    [2, 52, 77],
    [1, 71, 60],
    [1, 17, 56],
    [2, 12, 43],
    [1, 7, 49],
    [2, 159, 173],
    [2, 182, 181],
    [2, 270, 179],
    [1, 242, 183],
    [1, 264, 157],
    [2, 179, 136],
    [1, 185, 124],
    [1, 202, 129],
    [2, 231, 19],
    [1, 235, 53],
    [1, 218, 39],
    [2, 272, 114],
    [1, 298, 111],
    [2, 108, 44],
    [1, 145, 48],
    [2, 93, 108],
    [1, 76, 129],
    [1, 199, 208],
    [1, 226, 196],
    [1, 205, 158],
    [1, 213, 144],
])

hospitals = np.array([
    [2, 197, 172],
    [2, 192, 170],
    [1, 193, 175],
    [2, 210, 189],
    [1, 216, 167],
    [1, 194, 186],
    [1, 205, 173],
    [2, 188, 174],
    [1, 195, 153],
    [1, 200, 171],
    [2, 197, 172],
    [2, 198, 170],
    [2, 198, 171],
    [1, 195, 168],
    [1, 198, 171],
    [1, 196, 173],
    [2, 125, 192],
    [1, 126, 191],
    [1, 125, 192],
    [2, 126, 193],
    [2, 125, 192],
    [2, 159, 173],
    [1, 160, 173],
    [2, 182, 181],
    [1, 181, 181],
    [1, 166, 164],
    [1, 172, 197],
    [1, 170, 156],
    [1, 141, 194],
    [1, 156, 201],
    [2, 128, 189],
    [2, 139, 185],
    [1, 116, 176],
    [2, 114, 123],
    [2, 112, 123],
    [1, 115, 126],
    [1, 117, 124],
    [1, 114, 125],
    [1, 113, 124],
    [1, 112, 125],
    [1, 126, 103],
    [1, 149, 93],
    [1, 124, 84],
    [1, 104, 107],
    [1, 96, 138],
    [1, 131, 137],
    [1, 136, 148],
    [1, 142, 123],
    [2, 99, 146],
    [1, 99, 146],
    [2, 93, 108],
    [1, 86, 122],
    [1, 88, 92],
    [1, 98, 113],
    [2, 52, 77],
    [1, 52, 76],
    [1, 51, 77],
    [1, 53, 76],
    [1, 53, 76],
    [1, 50, 128],
    [1, 35, 71],
    [1, 17, 56],
    [1, 23, 60],
    [1, 46, 91],
    [2, 71, 60],
    [2, 12, 43],
    [1, 13, 44],
    [1, 14, 44],
    [1, 15, 45],
    [1, 7, 49],
    [1, 19, 66],
    [2, 179, 136],
    [1, 168, 144],
    [1, 162, 129],
    [1, 179, 136],
    [2, 195, 149],
    [1, 192, 147],
    [1, 182, 120],
    [1, 185, 124],
    [1, 202, 129],
    [1, 213, 144],
    [1, 204, 114],
    [2, 270, 179],
    [1, 270, 178],
    [1, 272, 177],
    [1, 242, 183],
    [1, 264, 157],
    [1, 232, 158],
    [1, 256, 140],
    [1, 275, 164],
    [1, 259, 173],
    [1, 226, 196],
    [2, 241, 203],
    [1, 242, 203],
    [2, 272, 114],
    [1, 298, 111],
    [1, 310, 73],
    [1, 326, 114],
    [1, 247, 100],
    [2, 235, 53],
    [2, 231, 19],
    [1, 233, 4],
    [1, 218, 39],
    [1, 230, 19],
    [1, 230, 18],
    [1, 218, 39],
    [1, 230, 19],
    [2, 108, 44],
    [1, 121, 49],
    [1, 93, 42],
    [1, 145, 48],
    [1, 159, 53],
    [2, 120, 152],
    [1, 121, 152],
    [2, 150, 162],
    [1, 130, 174],
    [1, 154, 156],
    [1, 190, 159],
    [1, 189, 162],
    [1, 251, 181],
])
sizes = hospitals[:, 0]
x = hospitals[:, 1]
y = hospitals[:, 2]

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

plt.title("Hospitals on Reduced Swiss Matrix")
plt.xlabel("Reduced matrix x-coordinate")
plt.ylabel("Reduced matrix y-coordinate")

plt.xlim(0, matrix.shape[1])
plt.ylim(0, matrix.shape[0])

plt.grid(color="lightblue", linewidth=0.3, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()