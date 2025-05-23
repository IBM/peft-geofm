# Copyright contributors to the PEFT-GeoFM project

from collections.abc import Sequence

import albumentations
import torch
from torch.utils.data import DataLoader
from torchgeo.datamodules import BaseDataModule

from peft_geofm.datamodules.utils import (
    Normalize,
    clay_collate_fn,
    wrap_in_compose_is_list,
)
from peft_geofm.datasets import MSACropType
from peft_geofm.datasets.utils import allowed_metadata

MEANS = {
    "COASTAL_AEROSOL": 12.739611,
    "BLUE": 16.526744,
    "GREEN": 26.636417,
    "RED": 36.696639,
    "RED_EDGE_1": 46.388679,
    "RED_EDGE_2": 58.281453,
    "RED_EDGE_3": 63.575819,
    "NIR_BROAD": 68.1836,
    "NIR_NARROW": 69.142591,
    "WATER_VAPOR": 69.904566,
    "SWIR_1": 83.626811,
    "SWIR_2": 65.767679,
    "CLOUD_PROBABILITY": 0.0,
}

STDS = {
    "COASTAL_AEROSOL": 7.492811526301659,
    "BLUE": 9.329547939662671,
    "GREEN": 12.674537246073758,
    "RED": 19.421922023931593,
    "RED_EDGE_1": 19.487411106531287,
    "RED_EDGE_2": 19.959174612412983,
    "RED_EDGE_3": 21.53805760692545,
    "NIR_BROAD": 23.05077775347288,
    "NIR_NARROW": 22.329695761624677,
    "WATER_VAPOR": 21.877766438821954,
    "SWIR_1": 28.14418826277069,
    "SWIR_2": 27.2346215312965,
    "CLOUD_PROBABILITY": 0.0,
}


class MSACropTypeDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        data_root: str,
        use_metadata: allowed_metadata,
        partition: str = "default",
        bands: Sequence[str] = MSACropType.all_band_names,
        train_transform: albumentations.Compose
        | None
        | list[albumentations.BasicTransform] = None,
        val_transform: albumentations.Compose
        | None
        | list[albumentations.BasicTransform] = None,
        test_transform: albumentations.Compose
        | None
        | list[albumentations.BasicTransform] = None,
        drop_last: bool = True,
    ) -> None:
        self.dataset_class: type[MSACropType]
        super().__init__(MSACropType, batch_size, num_workers)
        self.data_root = data_root
        self.use_metadata = use_metadata
        self.partition = partition
        self.bands = bands

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)

        self.drop_last = drop_last

        means = [MEANS[b] for b in bands]
        stds = [STDS[b] for b in bands]
        self.aug = Normalize(means, stds)

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train",
                data_root=self.data_root,
                use_metadata=self.use_metadata,
                bands=self.bands,
                transform=self.train_transform,
                partition=self.partition,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                use_metadata=self.use_metadata,
                bands=self.bands,
                transform=self.val_transform,
                partition=self.partition,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test",
                data_root=self.data_root,
                use_metadata=self.use_metadata,
                bands=self.bands,
                transform=self.test_transform,
                partition=self.partition,
            )
        if stage in ["predict"]:
            self.predict_dataset = self.dataset_class(
                split="test",
                data_root=self.data_root,
                use_metadata=self.use_metadata,
                bands=self.bands,
                transform=self.test_transform,
                partition=self.partition,
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, torch.Tensor]]:
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

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self._dataloader_factory("train")

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self._dataloader_factory("val")

    def test_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self._dataloader_factory("test")

    def predict_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self._dataloader_factory("predict")
