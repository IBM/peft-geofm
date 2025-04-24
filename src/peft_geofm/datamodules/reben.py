# Copyright contributors to the PEFT-GeoFM project

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import albumentations
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchgeo.datamodules import BaseDataModule

from peft_geofm.datamodules.utils import (
    Normalize,
    NormalizeAndPad,
    clay_collate_fn,
    wrap_in_compose_is_list,
)
from peft_geofm.datasets import ReBEN
from peft_geofm.datasets.utils import allowed_metadata

# taken from https://github.com/lhackel-tub/ConfigILM/blob/main/configilm/extra/BENv2_utils.py 120_nearest  # noqa: E501
MEANS = {
    "COASTAL_AEROSOL": 361.0767822265625,
    "BLUE": 438.3720703125,
    "GREEN": 614.0556640625,
    "RED": 588.4096069335938,
    "RED_EDGE_1": 942.8433227539062,
    "RED_EDGE_2": 1769.931640625,
    "RED_EDGE_3": 2049.551513671875,
    "NIR_BROAD": 2193.2919921875,
    "NIR_NARROW": 2235.556640625,
    "WATER_VAPOR": 2241.455322265625,
    "SWIR_1": 1568.226806640625,
    "SWIR_2": 997.7324829101562,
    # "VH": -19.352558135986328,
    # "VV": -12.643863677978516,
}

STDS = {
    "COASTAL_AEROSOL": 575.0687255859375,
    "BLUE": 607.02685546875,
    "GREEN": 603.2968139648438,
    "RED": 684.56884765625,
    "RED_EDGE_1": 738.4326782226562,
    "RED_EDGE_2": 1100.4560546875,
    "RED_EDGE_3": 1275.805419921875,
    "NIR_BROAD": 1369.3717041015625,
    "NIR_NARROW": 1356.5440673828125,
    "WATER_VAPOR": 1316.393310546875,
    "SWIR_1": 1070.1612548828125,
    "SWIR_2": 813.5276489257812,
    # "VH": 5.590505599975586,
    # "VV": 5.133493900299072,
}


class ReBENDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        data_root: str,
        use_metadata: allowed_metadata,
        label: Literal["classification", "segmentation"] = "segmentation",
        train_split: Path | None = None,
        val_split: Path | None = None,
        test_splits: list[Path] | None = None,
        predict_split: Path | None = None,
        bands: Sequence[str] = ReBEN.all_band_names,
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
        no_data_replace: float = 0,
        no_label_replace: int = -1,
        drop_last: bool = True,
        check_files: bool = True,
    ):
        self.dataset_class: type[ReBEN]
        super().__init__(ReBEN, batch_size, num_workers)
        self.data_root = data_root
        self.use_metadata = use_metadata
        self.label = label
        self.train_split = train_split
        self.val_split = val_split
        self.test_splits = test_splits
        self.predict_split = predict_split
        self.bands = bands
        self.constant_scale = constant_scale
        self.check_files = check_files

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)

        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.drop_last = drop_last

        self.test_datasets: list[Dataset[dict[str, Tensor]]] | None = None

        means = [MEANS[b] for b in bands]
        stds = [STDS[b] for b in bands]
        # We always pad to 8 pixels, because:
        # prithvi: 120 is not divisible by 16
        # clay: 120 is divisible by 8 but with UNet it is not compatible
        # (n_patches = 15 which is odd)
        # decur: the output has odd dimensions making it not compatible with UNet
        self.aug = NormalizeAndPad(
            Normalize(means, stds),
            (0, 8, 0, 8),
            "reflect",
            self.no_label_replace,
        )

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            assert self.train_split is not None
            self.train_dataset = self.dataset_class(
                self.data_root,
                self.use_metadata,
                label=self.label,
                split_file=self.train_split,
                bands=self.bands,
                transform=self.train_transform,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                check_files=self.check_files,
            )
        if stage in ["fit", "validate"]:
            assert self.val_split is not None
            self.val_dataset = self.dataset_class(
                self.data_root,
                self.use_metadata,
                label=self.label,
                split_file=self.val_split,
                bands=self.bands,
                transform=self.val_transform,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                check_files=self.check_files,
            )
        if stage in ["test"]:
            self.test_datasets = []
            assert self.test_splits is not None
            for test_split in self.test_splits:
                self.test_datasets.append(
                    self.dataset_class(
                        self.data_root,
                        self.use_metadata,
                        label=self.label,
                        split_file=test_split,
                        bands=self.bands,
                        transform=self.test_transform,
                        constant_scale=self.constant_scale,
                        no_data_replace=self.no_data_replace,
                        no_label_replace=self.no_label_replace,
                        check_files=self.check_files,
                    )
                )
        if stage in ["predict"]:
            assert self.predict_split is not None
            self.predict_dataset = self.dataset_class(
                self.data_root,
                self.use_metadata,
                label=self.label,
                split_file=self.predict_split,
                bands=self.bands,
                transform=self.test_transform,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                check_files=self.check_files,
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
