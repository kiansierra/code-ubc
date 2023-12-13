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

import wandb
from ubc import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    PROJECT_NAME,
    AugmentationDataset,
    build_augmentations,
    get_checkpoint_folder,
    idx2label,
    label2idx,
)

ROOT_DIR = Path("../input/UBC-OCEAN/")
PROCESSED_DIR = Path("../input/UBC-OCEAN-PROCESSED/")


def parser():
    parser = argparse.ArgumentParser(description="validation")
    parser.add_argument("--checkpoint-id", type=str, help="Checkpoint ID to validated", required=True)
    return parser.parse_args()


def aggregate_predictions(predictions: Dict[str, torch.Tensor]) -> pd.DataFrame:
    data = {k: predictions["probs"][:, v] for k, v in label2idx.items()}
    data["image_id"] = predictions["image_id"]
    data = pd.DataFrame(data).groupby("image_id", as_index=False).mean()
    data["pred"] = data[label2idx.keys()].values.argmax(axis=1)
    data["score"] = data[label2idx.keys()].values.max(axis=1)
    return data


def validate(checkpoint_id: str) -> None:
    torch.set_float32_matmul_precision("medium")
    run = wandb.init(job_type="validate", project=PROJECT_NAME)
    ckpt_artifact_dir = get_checkpoint_folder(checkpoint_id, run)
    config = OmegaConf.load(os.path.join(ckpt_artifact_dir, "config.yaml"))
    dataset_builder = DATASET_REGISTRY.get(config.dataframe.loader)
    _, valid_df = dataset_builder(run, config.dataframe)
    valid_ds = AugmentationDataset(valid_df, augmentation=build_augmentations(config.augmentations.valid))
    valid_dataloader = DataLoader(valid_ds, **config.dataloader.val_dataloader)
    config.model.pretrained = False
    builder = MODEL_REGISTRY.get(config.model.entrypoint)
    checkpoint_path = f"{ckpt_artifact_dir}/best.ckpt"
    model = builder.load_from_checkpoint(checkpoint_path, config=config, strict=False)
    model = model.eval()
    trainer = pl.Trainer(devices=1)
    predictions = trainer.predict(model, valid_dataloader)
    predictions = {k: torch.cat([elem[k] for elem in predictions], dim=0) for k in predictions[0].keys()}
    data = aggregate_predictions(predictions)
    valid_labels = valid_df.groupby("image_id", as_index=False).first()
    valid_labels["label"] = valid_labels["label"].map(label2idx).tolist()
    data = data.merge(valid_labels, on="image_id", how="left")
    data_tma = data.query("is_tma")
    data_wsi = data.query("not is_tma")
    bac_global = balanced_accuracy_score(data["label"], data["pred"])
    bac_wsi = balanced_accuracy_score(data_wsi["label"], data_wsi["pred"])
    bac_tma = balanced_accuracy_score(data_tma["label"], data_tma["pred"])
    bac_balanced = (bac_wsi + bac_tma) / 2
    logger.info(f"{bac_global=:.4f} for {checkpoint_id}")
    logger.info(f"{bac_wsi=:.4f} for {checkpoint_id}")
    logger.info(f"{bac_tma=:.4f} for {checkpoint_id}")
    logger.info(f"{bac_balanced=:.4f} for {checkpoint_id}")
    bac_dict = {"bac_global": bac_global, "bac_wsi": bac_wsi, "bac_tma": bac_tma, "bac_balanced": bac_balanced}
    run.log(bac_dict)
    data["label"] = data["label"].map(idx2label).tolist()
    data["pred"] = data["pred"].map(idx2label).tolist()
    table = wandb.Table(dataframe=data)
    artifact = wandb.Artifact(f"validation-{checkpoint_id}", type="validation", metadata=bac_dict)
    artifact.add(table, "validation-table")
    run.log_artifact(artifact, f"validation-{checkpoint_id}")
    run.finish()


if __name__ == "__main__":
    args = parser()
    validate(checkpoint_id=args.checkpoint_id)  # pylint: disable=no-value-for-parameter
    print("Done!")
