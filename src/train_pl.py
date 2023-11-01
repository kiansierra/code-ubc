from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ubc import (AugmentationDataset, TimmModel, get_train_transforms,
                 get_valid_transforms)

ROOT_DIR = Path("../input/UBC-OCEAN/")


def get_path(row):
    if row["is_tma"]:
        return str(ROOT_DIR / "train_images" / f"{row['image_id']}.png")
    return str(ROOT_DIR / "train_thumbnails" / f"{row['image_id']}_thumbnail.png")


@hydra.main(config_path="ubc/configs", config_name="config", version_base=None)
def train(config: DictConfig) -> None:
    df = pd.read_parquet(ROOT_DIR / "train.parquet")
    df["path"] = df.apply(get_path, axis=1)
    train_df = df[df["fold"] != config["fold"]].reset_index(drop=True)
    valid_df = df[df["fold"] == config["fold"]].reset_index(drop=True)
    train_ds = AugmentationDataset(train_df, augmentation=get_train_transforms(config))
    valid_ds = AugmentationDataset(valid_df, augmentation=get_valid_transforms(config))
    train_dataloader = DataLoader(train_ds, **config.dataloader.tr_dataloader)
    valid_dataloader = DataLoader(valid_ds, **config.dataloader.val_dataloader)
    model = TimmModel(config)
    trainer = pl.Trainer(**config["trainer"])
    trainer.fit(model, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter
    print("Done!")
