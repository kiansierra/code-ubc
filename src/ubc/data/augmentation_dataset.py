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
        output = {"image": image, "label": label2idx[row["label"]]}
        return output
