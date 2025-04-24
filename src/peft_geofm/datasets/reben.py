# Copyright contributors to the PEFT-GeoFM project

import os
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import albumentations
import numpy as np
import pandas as pd
import torch
from pyproj import CRS, Transformer
from terratorch.datasets.utils import (
    default_transform,
    validate_bands,
)
from torch.utils.data import Dataset
from tqdm import tqdm
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


class ReBEN(Dataset[dict[str, Any]]):
    all_band_names = (
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "WATER_VAPOR",
        "SWIR_1",
        "SWIR_2",
    )

    rgb_bands = ("RED", "GREEN", "BLUE")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    num_classes = 19

    class_names = [
        "Agro-forestry areas",
        "Arable land",
        "Beaches, dunes, sands",
        "Broad-leaved forest",
        "Coastal wetlands",
        "Complex cultivation patterns",
        "Coniferous forest",
        "Industrial or commercial units",
        "Inland waters",
        "Inland wetlands",
        "Land principally occupied by agriculture, with significant areas of natural vegetation",  # noqa: E501
        "Marine waters",
        "Mixed forest",
        "Moors, heathland and sclerophyllous vegetation",
        "Natural grassland and sparsely vegetated areas",
        "Pastures",
        "Permanent crops",
        "Transitional woodland, shrub",
        "Urban fabric",
    ]
    mask_val_to_label = {
        111: "Urban fabric",
        112: "Urban fabric",
        121: "Industrial or commercial units",
        122: "Unlabeled",
        123: "Unlabeled",
        124: "Unlabeled",
        131: "Unlabeled",
        132: "Unlabeled",
        133: "Unlabeled",
        141: "Unlabeled",
        142: "Unlabeled",
        211: "Arable land",
        212: "Arable land",
        213: "Arable land",
        221: "Permanent crops",
        222: "Permanent crops",
        223: "Permanent crops",
        231: "Pastures",
        241: "Permanent crops",
        242: "Complex cultivation patterns",
        243: "Land principally occupied by agriculture, with significant areas of natural vegetation",  # noqa: E501
        244: "Agro-forestry areas",
        311: "Broad-leaved forest",
        312: "Coniferous forest",
        313: "Mixed forest",
        321: "Natural grassland and sparsely vegetated areas",
        322: "Moors, heathland and sclerophyllous vegetation",
        323: "Moors, heathland and sclerophyllous vegetation",
        324: "Transitional woodland, shrub",
        331: "Beaches, dunes, sands",
        332: "Unlabeled",
        333: "Natural grassland and sparsely vegetated areas",
        334: "Unlabeled",
        335: "Unlabeled",
        411: "Inland wetlands",
        412: "Inland wetlands",
        421: "Coastal wetlands",
        422: "Coastal wetlands",
        423: "Unlabeled",
        511: "Inland waters",
        512: "Inland waters",
        521: "Marine waters",
        522: "Marine waters",
        523: "Marine waters",
        999: "Unlabeled",
    }

    # dictionary comprehension does not work here????
    _mask_val_to_ohe_index: dict[int, int] = {}
    for k, v in mask_val_to_label.items():
        if v != "Unlabeled":
            _mask_val_to_ohe_index[k] = class_names.index(v)
        else:
            _mask_val_to_ohe_index[k] = -1

    _max_key = max(_mask_val_to_ohe_index.keys())
    _lookup_array = np.zeros(_max_key + 1, dtype=int)
    for a, b in _mask_val_to_ohe_index.items():
        _lookup_array[a] = b

    @classmethod
    def reference_map_to_ohe_index(cls, mask: np.ndarray) -> np.ndarray:
        if not set(np.unique(mask)) <= set(cls._mask_val_to_ohe_index.keys()):
            raise ValueError("Invalid mask values")
        return cls._lookup_array[mask]  # type: ignore

    def __init__(
        self,
        data_root: str,
        use_metadata: allowed_metadata,
        split_file: Path,
        label: Literal["classification", "segmentation"] = "segmentation",
        bands: Sequence[str] = BAND_SETS["all"],
        transform: albumentations.Compose | None = None,
        constant_scale: float = 1,
        no_data_replace: float = 0,
        no_label_replace: int = -1,
        check_files: bool = True,
    ) -> None:
        super().__init__()
        validate_bands(bands, self.all_band_names)
        self.bands = bands
        self.band_indices = np.asarray([self.all_band_names.index(b) for b in bands])
        self.constant_scale = constant_scale
        self.data_root = Path(data_root)
        self.use_metadata = use_metadata
        self.label = label
        self.masks_path = self.data_root / "Reference_Maps"

        metadata = pd.read_parquet(self.data_root / "metadata.parquet")
        dummies = pd.get_dummies(metadata.explode("labels")["labels"])
        self.metadata = metadata.join(dummies.groupby(dummies.index).sum())
        self.metadata = self.metadata.set_index("patch_id", verify_integrity=True)

        assert split_file is not None
        with open(split_file) as f:
            split = f.readlines()
        valid_files = sorted([rf"{substring.strip()}" for substring in split])
        self.image_files = [
            self.data_root / "S2_merged" / f"{f}.tif" for f in valid_files
        ]
        if check_files:
            print("Checking all files exist...")
            for file in tqdm(self.image_files):
                assert file.is_file(), f"File {file} does not exist."
                mask_path = self._get_mask_path_from_filename(str(file))
                assert Path(mask_path).is_file(), f"File {mask_path} does not exist."

        if split_file is not None and len(valid_files) != len(self.image_files):
            raise ValueError("Check files and split files.")
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace

        # If no transform is given, apply only to transform to torch tensor
        self.transform = transform if transform else default_transform

    def __len__(self) -> int:
        return len(self.image_files)

    def _get_date(self, index: int) -> datetime:
        file_name = self.image_files[index]
        filename_regex = r"S2[AB]_MSIL2A_(?P<date>[0-9]{8})T(?P<time>[0-9]{6})_N9999_R[0-9]{3}_T[0-9A-Z]{5}_[0-9]{2}_[0-9]{2}.tif"  # noqa: E501
        match = re.fullmatch(filename_regex, os.path.basename(file_name))
        assert match is not None
        date = datetime.strptime(
            f"{match.group('date')} {match.group('time')}", "%Y%m%d %H%M%S"
        )
        return date

    def _get_labels(self, index: int) -> torch.Tensor:
        row = self.metadata.loc[self.image_files[index].stem]
        return torch.Tensor(row.loc[ReBEN.class_names].values.astype(np.float32))

    def _get_mask_path_from_filename(self, filename: str | Path) -> str:
        img_name = os.path.basename(filename)
        img_name_tile = "_".join(img_name.split("_")[0:6])
        h_order = img_name.split("_")[6]
        v_order = img_name.split("_")[7].split(".")[0]
        mask_path = (
            self.masks_path
            / img_name_tile
            / f"{img_name_tile}_{h_order}_{v_order}"
            / f"{img_name_tile}_{h_order}_{v_order}_reference_map.tif"
        )
        return str(mask_path)

    def get_mask_path(self, index: int) -> str:
        return self._get_mask_path_from_filename(self.image_files[index])

    def __getitem__(self, index: int) -> dict[str, Any]:
        rio_image = load_file(self.image_files[index], nan_replace=self.no_data_replace)

        # to channels last
        image = rio_image.to_numpy()
        image = np.moveaxis(image, 0, -1)

        # filter bands
        image = image[..., self.band_indices]

        output: dict[str, Any] = {
            "image": image.astype(np.float32) * self.constant_scale,
        }
        if self.label == "segmentation":
            mask = (
                load_file(self.get_mask_path(index), nan_replace=self.no_label_replace)
                .to_numpy()[0]
                .astype(np.int32)
            )
            output["mask"] = self.reference_map_to_ohe_index(mask)
        if self.transform:
            output = self.transform(**output)
        if self.label == "classification":
            output["label"] = self._get_labels(index)

        if self.use_metadata.startswith("clay"):
            chosen_metadata = self.use_metadata.split("_")[1]
            if "t" in chosen_metadata:
                output["time"] = normalize_timestamp_clay(self._get_date(index))
            if "l" in chosen_metadata:
                output["latlon"] = normalize_latlon_clay(*_get_lat_lon(rio_image))
            if "g" in chosen_metadata:
                output["gsd"] = 10
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
        output["filename"] = str(self.image_files[index])

        return output
