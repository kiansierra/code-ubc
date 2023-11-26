from typing import Dict, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import label2idx


class AugmentationDataset(Dataset):
    """
    Dataset class for loading images and labels.
    Applies augmentation if provided.
    """

    def __init__(self, dataframe: pd.DataFrame, augmentation: Optional[A.Compose] = None) -> None:
        super().__init__()
        self.records = dataframe.to_dict("records")
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.records)

    def load_image(self, path: str) -> np.ndarray:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, index: int) -> Dict[str, np.ndarray | torch.Tensor]:
        row = self.records[index]
        image = self.load_image(row["path"])
        if self.augmentation:
            image = self.augmentation(image=image)["image"]
        output = {"image_id": row["image_id"], "image": image}
        if "label" in row:
            output["label"] = label2idx[row["label"]]
        if "is_tma" in row:
            output["is_tma"] = int(row["is_tma"])
        return output


class TileDataset(Dataset):
    """
    Dataset class for loading images as tiles and labels.
    Applies augmentation if provided.
    """

    def __init__(self, dataframe: pd.DataFrame, augmentation: Optional[A.Compose] = None) -> None:
        super().__init__()
        self.groups = dataframe.groupby("image_id")
        self.image_ids = list(self.groups.groups.keys())
        self.num_images = 8
        self.tile_size = 512
        if augmentation:
            self.augmentation = A.Compose(
                augmentation.transforms,
                additional_targets={f"image_{num}": "image" for num in range(self.num_images - 1)},
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def load_image(self, path: str) -> np.ndarray:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, index: int) -> Dict[str, np.ndarray | torch.Tensor]:
        image_id = self.image_ids[index]
        full_group = self.groups.get_group(image_id)
        group = full_group.sample(n=self.num_images)
        images = [self.load_image(row["path"]) for _, row in group.iterrows()]
        if self.augmentation:
            images_dict = {f"image_{num}": image for num, image in enumerate(images[1:])}
            images_dict["image"] = images[0]
            transformed_images = self.augmentation(**images_dict)
            images = [transformed_images["image"]] + [
                transformed_images[f"image_{num}"] for num in range(self.num_images - 1)
            ]
            images = torch.stack(images)
        output = {"image_id": image_id, "image": images}
        output["pos_x"] = group["i"].values // self.tile_size
        output["pos_y"] = group["j"].values // self.tile_size

        if "label" in group:
            output["label"] = label2idx[group.iloc[0]["label"]]
        if "is_tma" in group:
            output["is_tma"] = int(group.iloc[0]["is_tma"])
        return output
