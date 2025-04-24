# Copyright contributors to the PEFT-GeoFM project

from collections.abc import Iterable, Sequence
from typing import Any

import albumentations
import torch
from torch import nn
from torch.utils.data import default_collate


def wrap_in_compose_is_list(
    transform_list: albumentations.Compose | Iterable,
) -> albumentations.Compose:
    return (
        albumentations.Compose(transform_list)
        if isinstance(transform_list, Iterable)
        else transform_list
    )


def clay_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    elems_to_default_collate = [
        {k: v for k, v in elem.items() if k not in ["gsd", "waves"]} for elem in batch
    ]
    collated_batch: dict[str, Any] = default_collate(elems_to_default_collate)
    # Assume all gsd and waves are the same
    collated_batch["gsd"] = batch[0]["gsd"]
    collated_batch["waves"] = batch[0]["waves"]
    return collated_batch


class Normalize:
    def __init__(self, means: list[float], stds: list[float]):
        super().__init__()
        self.means = means
        self.stds = stds

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        image = batch["image"]
        means = torch.tensor(self.means, device=image.device).view(1, -1, 1, 1)
        stds = torch.tensor(self.stds, device=image.device).view(1, -1, 1, 1)
        batch["image"] = (image - means) / stds
        return batch


class NormalizeAndPad:
    def __init__(
        self,
        normalize: Normalize,
        pad: tuple[int, ...],
        mode: str,
        no_label_replace: int,
    ):
        self.normalize = normalize
        self.pad = pad
        self.mode = mode
        self.no_label_replace = no_label_replace

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = self.normalize(batch)
        batch["image"] = nn.functional.pad(batch["image"], pad=self.pad, mode=self.mode)
        if "mask" in batch:
            batch["mask"] = nn.functional.pad(
                batch["mask"],
                pad=self.pad,
                mode="constant",
                value=self.no_label_replace,
            )
        return batch
