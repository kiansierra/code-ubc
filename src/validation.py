import argparse
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from ubc import AugmentationDataset, TimmModel, get_valid_transforms, label2idx

ROOT_DIR = Path("../input/UBC-OCEAN/")
PROCESSED_DIR = Path("../input/UBC-OCEAN-PROCESSED/")

def parser():
    parser = argparse.ArgumentParser(description="validation")
    parser.add_argument("--folder", type=str, help="folder to validate", required=True)
    return parser.parse_args()


def aggregate_predictions(predictions: Dict[str, torch.Tensor]) -> pd.DataFrame:
    data = {k: predictions["probs"][:, v] for k, v in label2idx.items()}
    data["image_id"] = predictions["image_id"]
    data = pd.DataFrame(data).groupby("image_id", as_index=False).mean()
    data["pred"] = data[label2idx.keys()].values.argmax(axis=1)
    return data


def validate(folder: str) -> None:
    torch.set_float32_matmul_precision("medium")
    config = OmegaConf.load(os.path.join(folder, "config.yaml"))
    df = pd.read_parquet(ROOT_DIR / f"{config.dataset.name}.parquet")
    valid_df = df.query(f"fold== {config['fold']}").reset_index(drop=True)
    valid_df = valid_df.groupby('image_id').sample(50, replace=True).drop_duplicates('path').reset_index(drop=True)
    valid_ds = AugmentationDataset(valid_df, augmentation=get_valid_transforms(config))
    valid_dataloader = DataLoader(valid_ds, **config.dataloader.val_dataloader)
    config.model.pretrained = False
    model = TimmModel.load_from_checkpoint(os.path.join(folder, "last.ckpt"), config=config)
    trainer = pl.Trainer(devices=1)
    predictions = trainer.predict(model, valid_dataloader)
    predictions = {k: torch.cat([elem[k] for elem in predictions], dim=0) for k in predictions[0].keys()}
    data = aggregate_predictions(predictions)
    # preds = predictions['probs'].argmax(1)
    valid_labels = valid_df.groupby("image_id", as_index=False).first()
    valid_labels["label"] = valid_labels["label"].map(label2idx).tolist()

    data = data.merge(valid_labels, on="image_id", how="left")
    bac = balanced_accuracy_score(data["label"], data["pred"])
    logger.info(f"Balanced Accuracy: {bac:.4f} for {folder}")
    data_tma = data.query('is_tma')
    bac_tma = balanced_accuracy_score(data_tma["label"], data_tma["pred"])
    logger.info(f"Balanced Accuracy TMA: {bac_tma:.4f} for {folder}")


if __name__ == "__main__":
    args = parser()
    validate(folder=args.folder)  # pylint: disable=no-value-for-parameter
    print("Done!")
