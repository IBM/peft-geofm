# Copyright contributors to the PEFT-GeoFM project

import re
import warnings
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import albumentations
import h5py
import numpy as np
import torch
from terratorch.datasets import MBeninSmallHolderCashewsNonGeo

from peft_geofm.datasets.utils import (
    WAVELENGTHS,
    allowed_metadata,
    normalize_timestamp_clay,
    normalize_timestamp_prithvi,
)


class MCashewPlantation(MBeninSmallHolderCashewsNonGeo):
    def __init__(
        self,
        data_root: str,
        use_metadata: allowed_metadata,
        split: str = "train",
        bands: Sequence[str] = MBeninSmallHolderCashewsNonGeo.BAND_SETS["all"],
        transform: albumentations.Compose | None = None,
        partition: str = "default",
    ) -> None:
        super().__init__(data_root, split, bands, transform, partition, False)
        self.use_metadata = use_metadata
        self._warning_shown = False

    def _get_date(self, keys: h5py.File) -> datetime:
        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")

        date_str = None
        for key in keys:
            match = date_pattern.search(key)
            if match:
                date_str = match.group()
                break
        assert date_str is not None

        return datetime.strptime(date_str, "%Y-%m-%d")

    def __getitem__(self, index: int) -> dict[str, Any]:
        file_path = self.image_files[index]

        with h5py.File(file_path, "r") as h5file:
            h5_keys = sorted(h5file.keys())
            keys = np.array([key for key in h5_keys if key != "label"])[
                self.band_indices
            ]
            bands = [np.array(h5file[key]) for key in keys]

            image = np.stack(bands, axis=-1)
            capture_date = self._get_date(h5file)
            mask = np.array(h5file["label"])

        output = {"image": image.astype(np.float32), "mask": mask}

        if self.transform:
            output = self.transform(**output)
        output["mask"] = output["mask"].long()

        if self.use_metadata.startswith("clay"):
            chosen_metadata = self.use_metadata.split("_")[1]
            if "t" in chosen_metadata:
                output["time"] = normalize_timestamp_clay(capture_date)
            if "l" in chosen_metadata and not self._warning_shown:
                self._warning_shown = True
                warnings.warn("Location metadata not available for this dataset")
            if "g" in chosen_metadata:
                output["gsd"] = 10
            if "w" in chosen_metadata:
                output["waves"] = torch.tensor([WAVELENGTHS[b] for b in self.bands])
        elif self.use_metadata.startswith("prithvi"):
            chosen_metadata = self.use_metadata.split("_")[1]
            if "t" in chosen_metadata:
                output["temporal_coords"] = normalize_timestamp_prithvi(capture_date)
            if "l" in chosen_metadata and not self._warning_shown:
                self._warning_shown = True
                warnings.warn("Location metadata not available for this dataset")
        elif self.use_metadata == "no_metadata":
            pass
        else:
            raise ValueError(f"use_metadata should be one of {allowed_metadata}")
        # Add filename for predict
        output["filename"] = str(self.image_files[index])
        return output
