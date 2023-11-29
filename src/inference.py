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
from ubc import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    PROJECT_NAME,
    AugmentationDataset,
    TimmModel,
    get_valid_transforms,
    idx2label,
    label2idx,
)

import wandb

ROOT_DIR = Path("../input/UBC-OCEAN/")
PROCESSED_DIR = Path("../input/UBC-OCEAN-PROCESSED/")


def parser():
    parser = argparse.ArgumentParser(description="validation")
    parser.add_argument("--checkpoint-folder", type=str, help="Checkpoint folder to validated", required=True)
    parser.add_argument("--input-df-path", type=str, help="Path to dataframe to run inference", required=True)
    parser.add_argument("--output-df-path", type=str, help="Path to save inference dataframe", default='best', required=True)
    parser.add_argument("--checkpoint-variant", type=str, help="Checkpoint variant to validated", default='best')
    return parser.parse_args()


def aggregate_predictions(predictions: Dict[str, torch.Tensor]) -> pd.DataFrame:
    data = {k: predictions["probs"][:, v] for k, v in label2idx.items()}
    data["image_id"] = predictions["image_id"]
    data = pd.DataFrame(data).groupby("image_id", as_index=False).mean()
    data["pred"] = data[label2idx.keys()].values.argmax(axis=1)
    data["score"] = data[label2idx.keys()].values.max(axis=1)
    return data


def inference() -> None:
    args = parser()
    torch.set_float32_matmul_precision("medium")
    config = OmegaConf.load(os.path.join(args.checkpoint_folder, "config.yaml"))
    checkpoint_path = f"{args.checkpoint_folder}/{args.checkpoint_variant}.ckpt"
    df = pd.read_parquet(args.input_df_path)
    ds = AugmentationDataset(df, augmentation=get_valid_transforms(config))
    config.dataloader.val_dataloader.shuffle = False
    dataloader = DataLoader(ds, **config.dataloader.val_dataloader)
    config.model.pretrained = False
    builder = MODEL_REGISTRY.get(config.model.entrypoint)
    model = builder.load_from_checkpoint(checkpoint_path, config=config, strict=False)
    trainer = pl.Trainer(devices=1)
    predictions = trainer.predict(model, dataloader)
    predictions = {k: torch.cat([elem[k] for elem in predictions], dim=0) for k in predictions[0].keys()}
    probs = predictions.pop("probs")
    image_ids = predictions.pop("image_id")
    for k, v in label2idx.items():
        predictions[k] = probs[:, v]
    pred_df = df.join(pd.DataFrame(predictions))
    pred_df.to_parquet(args.output_df_path)



if __name__ == "__main__":
    
    inference()  # pylint: disable=no-value-for-parameter
    print("Done!")
