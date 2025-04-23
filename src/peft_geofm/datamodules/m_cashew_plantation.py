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
from peft_geofm.datasets import MCashewPlantation
from peft_geofm.datasets.utils import allowed_metadata

MEANS = {
    "COASTAL_AEROSOL": 520.1185302734375,
    "BLUE": 634.7583618164062,
    "GREEN": 892.461181640625,
    "RED": 880.7075805664062,
    "RED_EDGE_1": 1380.6409912109375,
    "RED_EDGE_2": 2233.432373046875,
    "RED_EDGE_3": 2549.379638671875,
    "NIR_BROAD": 2643.248046875,
    "NIR_NARROW": 2643.531982421875,
    "WATER_VAPOR": 2852.87451171875,
    "SWIR_1": 2463.933349609375,
    "SWIR_2": 1600.9207763671875,
    "CLOUD_PROBABILITY": 0.010281000286340714,
}

STDS = {
    "COASTAL_AEROSOL": 204.2023468017578,
    "BLUE": 227.25344848632812,
    "GREEN": 222.32545471191406,
    "RED": 350.47235107421875,
    "RED_EDGE_1": 280.6436767578125,
    "RED_EDGE_2": 373.7521057128906,
    "RED_EDGE_3": 449.9236145019531,
    "NIR_BROAD": 414.6498107910156,
    "NIR_NARROW": 415.1019592285156,
    "WATER_VAPOR": 413.8980407714844,
    "SWIR_1": 494.97430419921875,
    "SWIR_2": 514.4229736328125,
    "CLOUD_PROBABILITY": 0.3447800576686859,
}


class MCashewPlantationDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        data_root: str,
        use_metadata: allowed_metadata,
        partition: str = "default",
        bands: Sequence[str] = MCashewPlantation.all_band_names,
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
        self.dataset_class: type[MCashewPlantation]
        super().__init__(MCashewPlantation, batch_size, num_workers)
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
