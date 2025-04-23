import glob
import os
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import albumentations
import numpy as np
import torch
from pyproj import CRS, Transformer
from terratorch.datasets.utils import (
    default_transform,
    filter_valid_files,
    validate_bands,
)
from torch.utils.data import Dataset
from xarray import DataArray

from peft_geofm.datasets.utils import (
    WAVELENGTHS,
    allowed_metadata,
    load_file,
    normalize_latlon_clay,
    normalize_latlon_prithvi,
    normalize_timestamp_clay,
    normalize_timestamp_prithvi,
)


def _get_lat_lon(image: DataArray) -> tuple[float, float]:
    transformer = Transformer.from_crs(CRS(image.spatial_ref.crs_wkt), "EPSG:4326")
    center_lat, center_lon = transformer.transform(
        image.x[image.x.shape[0] // 2].item(), image.y[image.y.shape[0] // 2].item()
    )
    return center_lat, center_lon


class BurnScars(Dataset[dict[str, Any]]):
    all_band_names = (
        "BLUE",
        "GREEN",
        "RED",
        "NIR_NARROW",
        "SWIR_1",
        "SWIR_2",
    )

    rgb_bands = ("RED", "GREEN", "BLUE")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    num_classes = 2
    class_names = ["Not burned", "Burn scar"]

    def __init__(
        self,
        data_root: str,
        use_metadata: allowed_metadata,
        split_file: Path | None = None,
        bands: Sequence[str] = BAND_SETS["all"],
        transform: albumentations.Compose | None = None,
        constant_scale: float = 1,
        no_data_replace: float | None = 0,
        no_label_replace: int | None = -1,
    ) -> None:
        super().__init__()
        validate_bands(bands, self.all_band_names)
        self.bands = bands
        self.band_indices = np.asarray([self.all_band_names.index(b) for b in bands])
        self.constant_scale = constant_scale
        self.data_root = Path(data_root)
        self.use_metadata = use_metadata

        self.image_files = sorted(
            glob.glob(os.path.join(self.data_root, "*_merged.tif"))
        )
        self.segmentation_mask_files = sorted(
            glob.glob(os.path.join(self.data_root, "*.mask.tif"))
        )
        if split_file is not None:
            with open(split_file) as f:
                split = f.readlines()
            valid_files = {rf"{substring.strip()}" for substring in split}
            self.image_files = filter_valid_files(
                self.image_files,
                valid_files=valid_files,
                ignore_extensions=True,
                allow_substring=True,
            )
            self.segmentation_mask_files = filter_valid_files(
                self.segmentation_mask_files,
                valid_files=valid_files,
                ignore_extensions=True,
                allow_substring=True,
            )
        if len(self.image_files) != len(self.segmentation_mask_files) or (
            split_file is not None and len(valid_files) != len(self.image_files)
        ):
            raise ValueError("Check files and split files.")
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace

        # If no transform is given, apply only to transform to torch tensor
        self.transform = transform if transform else default_transform

    def __len__(self) -> int:
        return len(self.image_files)

    def _get_date(self, index: int) -> datetime:
        file_name = self.image_files[index]
        filename_regex = r"subsetted_512x512_HLS\.S30\.T[0-9A-Z]{5}\.(?P<date>[0-9]+)\.v1\.4_merged\.tif"  # noqa: E501
        match = re.match(filename_regex, os.path.basename(file_name))
        assert match is not None
        julian_date = match.group("date")
        date = datetime.strptime(julian_date, "%Y%j").replace(hour=12)
        # There is no hour so all of them are assumed at midnight, we use noon
        return date

    def __getitem__(self, index: int) -> dict[str, Any]:
        rio_image = load_file(self.image_files[index], nan_replace=self.no_data_replace)

        # to channels last
        image = rio_image.to_numpy()
        image = np.moveaxis(image, 0, -1)

        # filter bands
        image = image[..., self.band_indices]

        output = {
            "image": image.astype(np.float32) * self.constant_scale,
            "mask": load_file(
                self.segmentation_mask_files[index], nan_replace=self.no_label_replace
            ).to_numpy()[0],
        }
        if self.transform:
            output = self.transform(**output)
        output["mask"] = output["mask"].long()

        if self.use_metadata.startswith("clay"):
            chosen_metadata = self.use_metadata.split("_")[1]
            if "t" in chosen_metadata:
                output["time"] = normalize_timestamp_clay(self._get_date(index))
            if "l" in chosen_metadata:
                output["latlon"] = normalize_latlon_clay(*_get_lat_lon(rio_image))
            if "g" in chosen_metadata:
                output["gsd"] = 30
            if "w" in chosen_metadata:
                output["waves"] = torch.tensor([WAVELENGTHS[b] for b in self.bands])
        elif self.use_metadata.startswith("prithvi"):
            chosen_metadata = self.use_metadata.split("_")[1]
            if "t" in chosen_metadata:
                output["temporal_coords"] = normalize_timestamp_prithvi(
                    self._get_date(index)
                )
            if "l" in chosen_metadata:
                output["location_coords"] = normalize_latlon_prithvi(
                    *_get_lat_lon(rio_image)
                )
        elif self.use_metadata == "no_metadata":
            pass
        else:
            raise ValueError(f"use_metadata should be one of {allowed_metadata}")
        # Add filename for predict
        output["filename"] = self.image_files[index]
        return output
