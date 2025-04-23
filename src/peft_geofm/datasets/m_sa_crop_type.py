import warnings
from collections.abc import Sequence
from typing import Any

import albumentations
import torch
from terratorch.datasets import MSACropTypeNonGeo
from peft_geofm.datasets.utils import WAVELENGTHS, allowed_metadata


class MSACropType(MSACropTypeNonGeo):  # type: ignore
    def __init__(
        self,
        data_root: str,
        use_metadata: allowed_metadata,
        split: str = "train",
        bands: Sequence[str] = MSACropTypeNonGeo.BAND_SETS["all"],
        transform: albumentations.Compose | None = None,
        partition: str = "default",
    ) -> None:
        super().__init__(data_root, split, bands, transform, partition)
        self.use_metadata = use_metadata
        self._location_warning_shown = False
        self._date_warning_shown = False

    def __getitem__(self, index: int) -> dict[str, Any]:
        output: dict[str, Any] = super().__getitem__(index)

        if self.use_metadata.startswith("clay"):
            chosen_metadata = self.use_metadata.split("_")[1]
            if "t" in chosen_metadata and not self._date_warning_shown:
                self._date_warning_shown = True
                warnings.warn("Date metadata not available for this dataset")
            if "l" in chosen_metadata and not self._location_warning_shown:
                self._location_warning_shown = True
                warnings.warn("Location metadata not available for this dataset")
            if "g" in chosen_metadata:
                output["gsd"] = 10
            if "w" in chosen_metadata:
                output["waves"] = torch.tensor([WAVELENGTHS[b] for b in self.bands])
        elif self.use_metadata.startswith("prithvi"):
            chosen_metadata = self.use_metadata.split("_")[1]
            if "t" in chosen_metadata and not self._date_warning_shown:
                self._date_warning_shown = True
                warnings.warn("Date metadata not available for this dataset")
            if "l" in chosen_metadata and not self._location_warning_shown:
                self._location_warning_shown = True
                warnings.warn("Location metadata not available for this dataset")
        elif self.use_metadata == "no_metadata":
            pass
        else:
            raise ValueError(f"use_metadata should be one of {allowed_metadata}")
        # Add filename for predict
        output["filename"] = str(self.image_files[index])
        return output
