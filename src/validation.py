import os
from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from ubc import AugmentationDataset, TimmModel, get_train_transforms, get_valid_transforms, label2idx

ROOT_DIR = Path("../input/UBC-OCEAN/")


def get_path(row):
    if row["is_tma"]:
        return str(ROOT_DIR / "train_images" / f"{row['image_id']}.png")
    return str(ROOT_DIR / "train_thumbnails" / f"{row['image_id']}_thumbnail.png")

def validate(folder: str) -> None:
    config = OmegaConf.load(os.path.join(folder, "config.yaml"))
    df = pd.read_parquet(ROOT_DIR / "train.parquet")
    df["path"] = df.apply(get_path, axis=1)
    valid_df = df[df["fold"] == config["fold"]].reset_index(drop=True)
    labels = valid_df['label'].map(label2idx).tolist()
    valid_ds = AugmentationDataset(valid_df, augmentation=get_valid_transforms(config))
    
    valid_dataloader = DataLoader(valid_ds, **config.dataloader.val_dataloader)
    config.model.pretrained = False
    # model = TimmModel(config)
    model = TimmModel.load_from_checkpoint(os.path.join(folder, "last.ckpt"), config=config)
    trainer = pl.Trainer(devices=1)
    predictions = trainer.predict(model, valid_dataloader)
    preds = torch.vstack(predictions).argmax(1)
    bac = balanced_accuracy_score(labels, preds)
    logger.info(f"Balanced Accuracy: {bac:.4f} for {folder}")




if __name__ == "__main__":
    validate(folder="../checkpoints/tf_efficientnetv2_s_in21ft1k/c3unvy7o")  # pylint: disable=no-value-for-parameter
    print("Done!")
