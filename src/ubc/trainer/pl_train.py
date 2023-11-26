import os
from pathlib import Path
from typing import List

import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader

import wandb
from ubc import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    PROJECT_NAME,
    AugmentationDataset,
    get_train_transforms,
    get_valid_transforms,
    label2idx,
    set_seed,
    upload_to_wandb,
)

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
    inverse_class_counts = 1 / class_counts
    class_weights = inverse_class_counts / inverse_class_counts.sum()
    return class_weights.tolist()


def load_checkpoint(config: DictConfig, model: pl.LightningModule, wandb_logger: WandbLogger) -> None:
    api = wandb.Api()
    run = api.run(f"{PROJECT_NAME}/{config.checkpoint_id}")
    ckpt_artifact_name = [
        artifact.name for artifact in run.logged_artifacts() if artifact.name.startswith(config.model.backbone)
    ][0]
    ckpt_artifact = wandb_logger.experiment.use_artifact(ckpt_artifact_name, type="model")
    ckpt_artifact_dir = ckpt_artifact.download()
    checkpoint_path = f"{ckpt_artifact_dir}/best.ckpt"
    state_dict = torch.load(checkpoint_path)["state_dict"]
    state_dict.pop("loss_fn.weight", None)
    model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded checkpoint: {config.checkpoint_id}")


def train_pl_run(config: DictConfig) -> None:
    # torch.set_float32_matmul_precision("medium")
    pl.seed_everything(config.seed, workers=True)
    set_seed(config.seed)
    config_container = OmegaConf.to_container(config, resolve=True)
    set_debug(config)
    tags = [
        config.model.backbone,
        config.model.entrypoint,
        config.dataset.artifact_name,
        f"img_size-{config.img_size}",
    ]
    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        dir=config.output_dir,
        tags=tags,
        config=config_container,
        offline=config.get("debug", False),
        job_type="train",
    )
    dataset_builder = DATASET_REGISTRY.get(config.dataset.loader)
    train_df, valid_df = dataset_builder(wandb_logger.experiment, config.dataset)
    train_ds = AugmentationDataset(train_df, augmentation=get_train_transforms(config))
    valid_ds = AugmentationDataset(valid_df, augmentation=get_valid_transforms(config))
    train_dataloader = DataLoader(train_ds, **config.dataloader.tr_dataloader)
    valid_dataloader = DataLoader(valid_ds, **config.dataloader.val_dataloader)
    weights = get_class_weights(train_df)
    use_weights = config.model.get("use_weights", False)
    model = MODEL_REGISTRY.get(config.model.entrypoint)(config, weights=weights if use_weights else None)
    config.max_steps = config.trainer.max_epochs * len(train_dataloader) // config.trainer.accumulate_grad_batches

    if config.get("checkpoint_id", False):
        load_checkpoint(config, model, wandb_logger)
        # logger.watch(model, log="gradients", log_freq=10)
    update_output_dir(config, wandb_logger)
    save_config(config)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # early_stopping = EarlyStopping(monitor=config["monitor"], patience=4, mode="max")
    checkpoint_callback = ModelCheckpoint(
        filename="best", dirpath=config.output_dir, monitor=config["monitor"], mode="max", save_last=True, save_top_k=1
    )
    trainer = pl.Trainer(**config["trainer"], logger=wandb_logger, callbacks=[lr_monitor, checkpoint_callback])
    trainer.fit(model, train_dataloader, valid_dataloader)
    run_id = wandb_logger.experiment.id
    rank_zero_only(upload_to_wandb)(config.output_dir, name=config.model.backbone)
    return run_id
