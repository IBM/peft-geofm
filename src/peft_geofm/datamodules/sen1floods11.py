# Copyright contributors to the PEFT-GeoFM project

from collections.abc import Sequence
from pathlib import Path

import albumentations
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchgeo.datamodules import BaseDataModule

from peft_geofm.datamodules.utils import (
    Normalize,
    clay_collate_fn,
    wrap_in_compose_is_list,
)
from peft_geofm.datasets import Sen1Floods11
from peft_geofm.datasets.utils import allowed_metadata

MEANS = {
    "COASTAL_AEROSOL": 0.16450718,
    "BLUE": 0.1412956,
    "GREEN": 0.13795798,
    "RED": 0.12353792,
    "RED_EDGE_1": 0.1481099,
    "RED_EDGE_2": 0.23991728,
    "RED_EDGE_3": 0.28587557,
    "NIR_BROAD": 0.26345379,
    "NIR_NARROW": 0.30902815,
    "WATER_VAPOR": 0.04911151,
    "CIRRUS": 0.00652506,
    "SWIR_1": 0.2044958,
    "SWIR_2": 0.11912015,
}

STDS = {
    "COASTAL_AEROSOL": 0.06977374,
    "BLUE": 0.07406382,
    "GREEN": 0.07370365,
    "RED": 0.08692279,
    "RED_EDGE_1": 0.07778555,
    "RED_EDGE_2": 0.09105416,
    "RED_EDGE_3": 0.10690993,
    "NIR_BROAD": 0.10096586,
    "NIR_NARROW": 0.11798815,
    "WATER_VAPOR": 0.03380113,
    "CIRRUS": 0.01463465,
    "SWIR_1": 0.09772074,
    "SWIR_2": 0.07659938,
}


class Sen1Floods11DataModule(BaseDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        data_root: str,
        use_metadata: allowed_metadata,
        train_split: Path | None = None,
        val_split: Path | None = None,
        test_splits: list[Path] | None = None,
        predict_split: Path | None = None,
        bands: Sequence[str] = Sen1Floods11.all_band_names,
        constant_scale: float = 0.0001,
        train_transform: albumentations.Compose
        | None
        | list[albumentations.BasicTransform] = None,
        val_transform: albumentations.Compose
        | None
        | list[albumentations.BasicTransform] = None,
        test_transform: albumentations.Compose
        | None
        | list[albumentations.BasicTransform] = None,
        no_data_replace: float = 0,
        no_label_replace: int = -1,
        drop_last: bool = True,
        # pad_ps_14: bool = False,
    ):
        self.dataset_class: type[Sen1Floods11]
        super().__init__(Sen1Floods11, batch_size, num_workers)
        self.data_root = data_root
        self.use_metadata = use_metadata
        self.train_split = train_split
        self.val_split = val_split
        self.test_splits = test_splits
        self.predict_split = predict_split
        self.bands = bands
        self.constant_scale = constant_scale

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)

        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.drop_last = drop_last

        self.test_datasets: list[Dataset[dict[str, Tensor]]] | None = None

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
            self.test_datasets = []
            assert self.test_splits is not None
            for test_split in self.test_splits:
                self.test_datasets.append(
                    self.dataset_class(
                        self.data_root,
                        self.use_metadata,
                        split_file=test_split,
                        bands=self.bands,
                        transform=self.test_transform,
                        constant_scale=self.constant_scale,
                        no_data_replace=self.no_data_replace,
                        no_label_replace=self.no_label_replace,
                    )
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

    def test_dataloader(self) -> list[DataLoader[dict[str, Tensor]]]:
        dataloaders: list[DataLoader[dict[str, Tensor]]] = []
        assert self.test_datasets is not None
        for dataset in self.test_datasets:
            batch_size = self._valid_attribute("test_batch_size", "batch_size")
            dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    drop_last=False,
                    collate_fn=clay_collate_fn
                    if self.use_metadata.startswith("clay")
                    else None,
                )
            )
        return dataloaders

    def predict_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return self._dataloader_factory("predict")
