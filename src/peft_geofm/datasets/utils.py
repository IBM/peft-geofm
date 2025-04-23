from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import rioxarray
import torch
from xarray import DataArray

allowed_metadata = Literal["clay_tlgw", "clay_gw", "prithvi_tl", "no_metadata"]


def normalize_timestamp_clay(date: pd.Timestamp | datetime) -> torch.Tensor:
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24
    return torch.Tensor([np.sin(week), np.cos(week), np.sin(hour), np.cos(hour)])


def normalize_timestamp_prithvi(date: datetime) -> torch.Tensor:
    day_of_year = date.timetuple().tm_yday
    year = date.year
    date_tensor = torch.Tensor(
        [[year, day_of_year - 1]]
    )  # 0-indexed and added time dimension
    return date_tensor


def normalize_latlon_clay(lat: float, lon: float) -> torch.Tensor:
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    return torch.Tensor([np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)])


def normalize_latlon_prithvi(lat: float, lon: float) -> torch.Tensor:
    return torch.Tensor([lat, lon])


def load_file(path: str | Path, nan_replace: int | float | None = None) -> DataArray:
    data = rioxarray.open_rasterio(path, masked=True)
    assert isinstance(data, DataArray)
    if nan_replace is not None:
        data = data.fillna(nan_replace)
    return data


WAVELENGTHS = {
    "COASTAL_AEROSOL": 0.443,
    "BLUE": 0.493,
    "GREEN": 0.56,
    "RED": 0.665,
    "RED_EDGE_1": 0.704,
    "RED_EDGE_2": 0.74,
    "RED_EDGE_3": 0.783,
    "NIR_BROAD": 0.842,
    "NIR_NARROW": 0.865,
    "WATER_VAPOR": 0.94,
    "CIRRUS": 1.375,
    "SWIR_1": 1.61,
    "SWIR_2": 2.19,
}
