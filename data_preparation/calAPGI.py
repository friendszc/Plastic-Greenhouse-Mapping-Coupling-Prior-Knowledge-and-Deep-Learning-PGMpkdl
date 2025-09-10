from pathlib import Path
import numpy as np
import rasterio

def compute_apgi(b1: np.ndarray, b4: np.ndarray, b8: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """Return APGI as *float32* with **NaN** where the denominator is zero.
    Assumes all bands are scaled by 10000 and rescales internally.
    """
    scale = 1 / 10000.0
    b1, b4, b8, b12 = b1 * scale, b4 * scale, b8 * scale, b12 * scale
    num = 100.0 * b1 * b4 * (2.0 * b8 - b4 - b12)
    den = 2.0 * b8 + b4 + b12
    with np.errstate(divide="ignore", invalid="ignore"):
        apgi = np.where(den != 0, num / den, np.nan)
    return apgi.astype(np.float32, copy=False)

def process_single_tif(in_file: Path, out_file: Path) -> None:
    """Compute APGI for one Sentinel‑2 GeoTIFF and write a new single‑band file."""
    with rasterio.open(in_file) as src:
        b1 = src.read(1).astype(np.float32)
        b4 = src.read(4).astype(np.float32)
        b8 = src.read(8).astype(np.float32)
        b12 = src.read(12).astype(np.float32)

        apgi = compute_apgi(b1, b4, b8, b12)

        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")

        out_file.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_file, "w", **profile) as dst:
            dst.write(apgi, 1)

def batch_process(in_path: Path, out_path: Path) -> None:
    """Recursively compute APGI for every ``*.tif`` under *in_path*."""
    for tif in in_path.glob("*.tif"):
        out_name = tif.stem[:1] + "_APGI.tif"
        out_file = out_path / out_name
        process_single_tif(tif, out_file)

if __name__ == "__main__":
    inPath = Path(__file__).parents[1].joinpath("Data", "S2")
    outPath = Path(__file__).parents[1].joinpath("Data", "APGI")

    print(f"Input directory : {inPath}")
    print(f"Output directory: {outPath}")

    batch_process(inPath, outPath)