import os
from pathlib import Path
from typing import List

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader

from ubc import (MODEL_REGISTRY, AugmentationDataset, get_train_transforms,
                 get_valid_transforms, label2idx, upload_to_wandb)

ROOT_DIR = Path("../input/UBC-OCEAN/")


@rank_zero_only
def update_output_dir(config: DictConfig, logger) -> DictConfig:
    config.output_dir = config.output_dir + "/" + logger.experiment.id


@rank_zero_only
def save_config(config: DictConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    OmegaConf.save(config, config.output_dir + "/config.yaml")


def set_debug(config: DictConfig):
    if config.get("debug", False):
        with open_dict(config):
            config.dataloader.tr_dataloader.num_workers = 0
            config.dataloader.val_dataloader.num_workers = 0
            config.dataloader.tr_dataloader.persistent_workers = False
            config.dataloader.val_dataloader.persistent_workers = False
            config.trainer.devices = 1
            config.trainer.fast_dev_run = True

def get_class_weights(df: pd.DataFrame) -> List[int]:
    class_counts = df["label"].map(label2idx).value_counts().sort_index()
    class_weights = class_counts / class_counts.sum() 
    return class_weights.tolist()

@hydra.main(config_path="ubc/configs", config_name="tf_efficientnetv2_s_in21ft1k", version_base=None)
def train(config: DictConfig) -> None:
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(config.seed, workers=True)
    df = pd.read_parquet(config.dataset.path)
    set_debug(config)
    train_df = df[df["fold"] != config["fold"]].reset_index(drop=True)
    valid_df = df[df["fold"] == config["fold"]].reset_index(drop=True)
    train_ds = AugmentationDataset(train_df, augmentation=get_train_transforms(config))
    valid_ds = AugmentationDataset(valid_df, augmentation=get_valid_transforms(config))
    train_dataloader = DataLoader(train_ds, **config.dataloader.tr_dataloader)
    valid_dataloader = DataLoader(valid_ds, **config.dataloader.val_dataloader)
    weights = get_class_weights(train_df)
    model = MODEL_REGISTRY.get(config.model.entrypoint)(config, weights=weights)
    config_container = OmegaConf.to_container(config, resolve=True)
    dataset_name = Path(config.dataset.path).parent.name
    tags = [config.model.backbone, dataset_name, f"img_size-{config.img_size}"]
    logger = WandbLogger(
        project="UBC-OCEAN",
        dir=config.output_dir,
        tags=tags,
        config=config_container,
        offline=config.get("debug", False),
        job_type="train",
    )
    if config.get("checkpoint_id", False):
        checkpoint_path = f"{config.output_dir}/{config.checkpoint_id}/last.ckpt"
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        logger.watch(model, log="gradients", log_freq=10)
    update_output_dir(config, logger)
    save_config(config)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(monitor=config["monitor"], patience=5, mode="max")
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.output_dir, monitor=config["monitor"], mode="max", save_last=True, save_top_k=2
    )
    trainer = pl.Trainer(
        **config["trainer"], logger=logger, callbacks=[lr_monitor, early_stopping, checkpoint_callback]
    )
    trainer.fit(model, train_dataloader, valid_dataloader)
    rank_zero_only(upload_to_wandb)(config.output_dir, name=config.model.backbone)


if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter
    print("Done!")
