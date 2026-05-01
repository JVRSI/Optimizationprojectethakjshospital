import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Create a 2D Gaussian kernel array.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def convolve_2d(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform a 2D convolution of the data with the given kernel.
    """
    from scipy.signal import convolve2d

    return convolve2d(data, kernel, mode="same", boundary="fill", fillvalue=0)


def downsample_sum(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Reduce the grid size by aggregating factor x factor cells into one cell.

    Example:
    - factor = 10 means every 10 x 10 block becomes one new cell.
    - Values are summed, so the total population is preserved as much as possible.
    """
    if factor <= 0:
        raise ValueError("Downsampling factor must be larger than 0.")

    rows, cols = data.shape

    new_rows = rows // factor
    new_cols = cols // factor

    trimmed = data[:new_rows * factor, :new_cols * factor]

    downsampled = trimmed.reshape(new_rows, factor, new_cols, factor).sum(axis=(1, 3))

    return downsampled


def create_convolved_heatmap(
    input_csv: Path,
    output_csv: Path,
    output_png: Path,
    sigma: float,
    kernel_size: int,
    factor: int,
    separator: str = ";",
) -> None:
    print(f"Sigma: {sigma}")
    print(f"Kernel size: {kernel_size} x {kernel_size}")
    print(f"Downsampling factor: {factor}")

    data = pd.read_csv(input_csv, sep=separator, header=None).values

    kernel = gaussian_kernel(size=kernel_size, sigma=sigma)
    convolved = convolve_2d(data, kernel)
    reduced = downsample_sum(convolved, factor=factor)

    pd.DataFrame(reduced).to_csv(output_csv, sep=separator, header=False, index=False)
    print(f"Reduced convolved matrix saved to: {output_csv.resolve()}")
    print(f"Output matrix shape: {reduced.shape[0]} rows x {reduced.shape[1]} columns")

    # Plot with zero values masked, so empty areas stay white
    reduced_masked = np.ma.masked_where(reduced == 0, reduced)

    positive_values = reduced[reduced > 0]

    vmax = np.percentile(positive_values, 99)
    cmap = plt.cm.Reds

    plt.imshow(
        reduced_masked,
        cmap=cmap,
        interpolation="nearest",
        origin="lower",
        vmin=0,
        vmax=vmax,
    )
    plt.colorbar()
    plt.title(
        f"Reduced Convolved STATPOP Heatmap, sigma={sigma}, kernel={kernel_size}x{kernel_size}, factor={factor}"
    )
    plt.savefig(output_png)
    plt.close()

    print(f"Original total BBTOT: {data.sum():.2f}")
    print(f"Convolved total BBTOT: {convolved.sum():.2f}")
    print(f"Reduced total BBTOT: {reduced.sum():.2f}")
    print("Note: If the original dimensions are not divisible by the factor, border cells are trimmed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a convolved heatmap from CSV data.")
    parser.add_argument("input_csv", type=Path, help="Input CSV file path")
    parser.add_argument("output_csv", type=Path, help="Output CSV file path")
    parser.add_argument("output_png", type=Path, help="Output PNG file path")
    parser.add_argument("--sigma", type=float, default=10.0, help="Sigma for Gaussian kernel")
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=51,
        help="Kernel size (must be odd)",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=10,
        help="Grid reduction factor. Example: 10 means every 10x10 cells become one cell. Default: 10",
    )

    args = parser.parse_args()

    if args.kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd, for example 11, 21, 31, 51.")

    if args.factor <= 0:
        raise ValueError("Factor must be larger than 0.")

    create_convolved_heatmap(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_png=args.output_png,
        sigma=args.sigma,
        kernel_size=args.kernel_size,
        factor=args.factor,
    )
