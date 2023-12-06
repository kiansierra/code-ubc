from typing import Dict, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils import Registry
from .constants import label2idx

DATASETS = Registry("datasets", Dataset)


@DATASETS.register()
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
        if "weight" in row:
            output["weight"] = row["weight"]
        return output

@DATASETS.register()
class TileDataset(Dataset):
    """
    Dataset class for loading images as tiles and labels.
    Applies augmentation if provided.
    """

    def __init__(self, dataframe: pd.DataFrame, augmentation: Optional[A.Compose] = None) -> None:
        super().__init__()
        self.dataframe = dataframe
        self.num_tiles = 6
        self.indexes = dataframe.query(f"i < max_i - {self.num_tiles} and j < max_j - {self.num_tiles}")[['image_id', 'component', 'i', 'j']].reset_index(drop=True)
        self.num_images = self.num_tiles ** 2
        self.tile_size = 512
        if augmentation:
            self.augmentation = A.Compose(
                augmentation.transforms,
                additional_targets={f"image_{num}": "image" for num in range(self.num_images - 1)},
            )
        else:
            self.augmentation = None

    def __len__(self) -> int:
        return len(self.indexes)

    def load_image(self, path: str) -> np.ndarray:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def get_sample_dataset(self, image_id:int, component:int, i:int, j:int) -> pd.DataFrame:
        sample_df = self.dataframe.query(f"image_id == {image_id} and component=={component} and i >= {i} and i < {i} + {self.num_tiles} and j >= {j} and j < {j} + {self.num_tiles}")
        return sample_df.reset_index(drop=True)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray | torch.Tensor]:
        index_row = self.indexes.loc[index]
        sample_df = self.get_sample_dataset(index_row["image_id"], index_row['component'], index_row["i"], index_row["j"])
        images = [self.load_image(path) for path in sample_df['path']]
        if self.augmentation:
            images_dict = {f"image_{num}": image for num, image in enumerate(images[1:])}
            images_dict["image"] = images[0]
            transformed_images = self.augmentation(**images_dict)
            images = [transformed_images["image"]] + [
                transformed_images[f"image_{num}"] for num in range(self.num_images - 1)
            ]
            images = torch.stack(images)
        output = {"image_id": index_row["image_id"], "image": images}
        output["pos_x"] = sample_df["i"].values - index_row["i"]
        output["pos_y"] = sample_df["j"].values - index_row["j"]
        if "weight" in sample_df:
            output["weight"] = sample_df["weight"].mean()
        if "label" in sample_df:
            output["label"] = label2idx[sample_df.iloc[0]["label"]]
        if "is_tma" in sample_df:
            output["is_tma"] = int(sample_df.iloc[0]["is_tma"])
        return output

