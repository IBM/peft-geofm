# Copyright contributors to the PEFT-GeoFM project

from collections.abc import Sequence
from pathlib import Path

import albumentations
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import BaseDataModule

from .utils import (
    Normalize,
    clay_collate_fn,
    wrap_in_compose_is_list,
)
from peft_geofm.datasets import BurnScars
from peft_geofm.datasets.utils import allowed_metadata

MEANS = {
    "BLUE": 0.0525628887,
    "GREEN": 0.0779808834,
    "RED": 0.0946640745,
    "NIR_NARROW": 0.2139334530,
    "SWIR_1": 0.2355762571,
    "SWIR_2": 0.1710367799,
}

STDS = {
    "BLUE": 0.0308036711,
    "GREEN": 0.0375606678,
    "RED": 0.0549106970,
    "NIR_NARROW": 0.0701380521,
    "SWIR_1": 0.0910611823,
    "SWIR_2": 0.0835518017,
}


class BurnScarsDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        data_root: str,
        use_metadata: allowed_metadata,
        train_split: Path | None = None,
        val_split: Path | None = None,
        test_split: Path | None = None,
        predict_split: Path | None = None,
        bands: Sequence[str] = BurnScars.all_band_names,
        constant_scale: float = 1,
        train_transform: albumentations.Compose
        | None
        | list[albumentations.BasicTransform] = None,
        val_transform: albumentations.Compose
        | None
        | list[albumentations.BasicTransform] = None,
        test_transform: albumentations.Compose
        | None
        | list[albumentations.BasicTransform] = None,
        no_data_replace: float | None = 0,
        no_label_replace: int | None = -1,
        drop_last: bool = True,
    ):
        self.dataset_class: type[BurnScars]
        super().__init__(BurnScars, batch_size, num_workers)
        self.data_root = data_root
        self.use_metadata = use_metadata
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.predict_split = predict_split
        self.bands = bands
        self.constant_scale = constant_scale

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)

        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.drop_last = drop_last

        means = [MEANS[b] for b in bands]
        stds = [STDS[b] for b in bands]
        self.aug = Normalize(means, stds)

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                self.data_root,
                self.use_metadata,
                split_file=self.train_split,
                bands=self.bands,
                transform=self.train_transform,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
            )
        if stage in ["fit", "validate"]:
            assert self.val_split is not None
            self.val_dataset = self.dataset_class(
                self.data_root,
                self.use_metadata,
                split_file=self.val_split,
                bands=self.bands,
                transform=self.val_transform,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
            )
        if stage in ["test"]:
            assert self.train_split is not None
            self.test_dataset = self.dataset_class(
                self.data_root,
                self.use_metadata,
                split_file=self.test_split,
                bands=self.bands,
                transform=self.test_transform,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
            )
        if stage in ["predict"]:
            assert self.predict_split is not None
            self.predict_dataset = self.dataset_class(
                self.data_root,
                self.use_metadata,
                split_file=self.predict_split,
                bands=self.bands,
                transform=self.test_transform,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            drop_last=split == "train" and self.drop_last,
            collate_fn=clay_collate_fn
            if self.use_metadata.startswith("clay")
            else None,
        )

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return self._dataloader_factory("train")

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return self._dataloader_factory("val")

    def test_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return self._dataloader_factory("test")

    def predict_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return self._dataloader_factory("predict")
