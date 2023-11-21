import os
from copy import deepcopy
from pathlib import Path
from typing import List

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
from ubc import (
    MODEL_REGISTRY,
    AugmentationDataset,
    get_train_transforms,
    get_valid_transforms,
    label2idx,
    upload_to_wandb,
)

ROOT_DIR = Path("../input/UBC-OCEAN/")


class Trainer:
    def __init__(self, accelerator, model, train_dataloader, valid_dataloader) -> None:
        self.accelerator = accelerator
        optimizer, scheduler = model.configure_optimizers()
        optimizer = optimizer[0]
        scheduler = scheduler["scheduler"]
        train_dataloader, valid_dataloader, optimizer, model = accelerator.prepare(
            train_dataloader, valid_dataloader, optimizer, model
        )
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model

    def train_epoch(self):
        for num, batch in enumerate(self.train_dataloader):
            self.model.train_step(batch, num)


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
    inverse_class_counts = 1 / class_counts
    class_weights = inverse_class_counts / inverse_class_counts.sum()
    return class_weights.tolist()


@hydra.main(config_path="ubc/configs", config_name="tf_efficientnetv2_s_in21ft1k", version_base=None)
def train_folds(config: DictConfig) -> None:
    if isinstance(config.fold, int):
        return train_img_size(config)
    for fold in config.fold:
        fold_config = deepcopy(config)
        fold_config.fold = fold
        train_img_size(fold_config)


def train_img_size(config: DictConfig) -> None:
    if isinstance(config.img_size, int):
        return train_once(config)
    new_ckpt = config.checkpoint_id
    for img_size in config.img_size:
        img_size_config = deepcopy(config)
        config.checkpoint_id = new_ckpt
        img_size_config.img_size = img_size
        new_ckpt = train_once(img_size_config)
        new_ckpt = Path(new_ckpt).name


def train_once(config: DictConfig) -> None:
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
    config.max_steps = config.trainer.max_epochs * len(train_dataloader) // config.trainer.accumulate_grad_batches
    config_container = OmegaConf.to_container(config, resolve=True)
    dataset_name = Path(config.dataset.path).parent.name
    tags = [config.model.backbone, dataset_name, f"img_size-{config.img_size}"]
    wandb_logger = WandbLogger(
        project="UBC-OCEAN",
        dir=config.output_dir,
        tags=tags,
        config=config_container,
        offline=config.get("debug", False),
        job_type="train",
    )
    if config.get("checkpoint_id", False):
        checkpoint_path = f"{config.output_dir}/{config.checkpoint_id}/last.ckpt"
        model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
        logger.debug(f"Loaded checkpoint {checkpoint_path}")
        wandb_logger.watch(model, log="gradients", log_freq=10)
    update_output_dir(config, wandb_logger)
    save_config(config)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(monitor=config["monitor"], patience=10, mode="max")
    checkpoint_callback = ModelCheckpoint(
        filename="best", dirpath=config.output_dir, monitor=config["monitor"], mode="max", save_last=True, save_top_k=1
    )
    trainer = pl.Trainer(
        **config["trainer"], logger=wandb_logger, callbacks=[lr_monitor, early_stopping, checkpoint_callback]
    )
    trainer.fit(model, train_dataloader, valid_dataloader)
    rank_zero_only(upload_to_wandb)(config.output_dir, name=config.model.backbone)
    return config.output_dir


if __name__ == "__main__":
    train_folds()  # pylint: disable=no-value-for-parameter
    print("Done!")
