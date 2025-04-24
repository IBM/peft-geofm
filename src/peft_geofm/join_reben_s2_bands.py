# Copyright contributors to the PEFT-GeoFM project

import os
from pathlib import Path

import rioxarray
import xarray
from jsonargparse import CLI
from tqdm import tqdm

band_suffixes = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]


def read_bands(folder_path: Path) -> list[xarray.DataArray]:
    chip_name = folder_path.name
    bands: list[xarray.DataArray] = []
    for suffix in band_suffixes:
        band_file_path = folder_path / f"{chip_name}_{suffix}.tif"
        if os.path.exists(band_file_path):
            bands.append(rioxarray.open_rasterio(band_file_path, masked=True))  # type: ignore
        else:
            print(f"Warning: {band_file_path} does not exist.")
    return bands


def interp_and_concat_bands(bands: list[xarray.DataArray]) -> xarray.DataArray:
    largest_index = 1
    interp_bands = []
    for band in bands:
        if band.isnull().any():
            raise ValueError  # to check for null values
        try:
            xarray.align(band, bands[largest_index], join="exact")
            interp_bands.append(band)
        except ValueError:
            new_band = band.interp_like(
                bands[largest_index],
                method="nearest",
                kwargs={"fill_value": "extrapolate"},
            )

            new_band.rio.write_transform(
                bands[largest_index].rio.transform(), inplace=True
            )
            interp_bands.append(new_band)

    return xarray.concat(
        interp_bands, "band", compat="identical", join="exact", coords="minimal"
    )


def join_s2_bands(files: list[str], root_dir: Path) -> None:
    s2_dir = root_dir / "BigEarthNet-S2"
    merged_folder = root_dir / "S2_merged"
    merged_folder.mkdir(exist_ok=True)
    chips_to_merge: list[str] = []
    for file in files:
        with open(file) as f:
            chips = f.readlines()
        chips = [f"{substring.strip()}" for substring in chips]
        chips_to_merge.extend(chips)
    skipped = 0
    for chip in tqdm(chips_to_merge):
        if (merged_folder / f"{chip}.tif").exists():
            skipped += 1
            continue
        chip_folder = s2_dir / chip[:-6] / chip
        bands = read_bands(chip_folder)
        interp_and_concat_bands(bands).rio.to_raster(merged_folder / f"{chip}.tif")
    print(f"Skipped {skipped} files.")


if __name__ == "__main__":
    CLI(join_s2_bands)
