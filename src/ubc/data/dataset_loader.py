from typing import Protocol, Tuple

import pandas as pd
from omegaconf import DictConfig

from wandb import wandb_sdk

from ..data.constants import label2idxmask
from ..utils import PROJECT_NAME, Registry


class DatasetLoaderProtocol(Protocol):
    def __call__(self, run: wandb_sdk.wandb_run.Run, config: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ...


class DatasetLoader(DatasetLoaderProtocol):
    def __call__(self, run: wandb_sdk.wandb_run.Run, config: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass


# DatasetLoader = Callable[[wandb_sdk.wandb_run.Run, DictConfig], Tuple[pd.DataFrame, pd.DataFrame]]

DATASET_REGISTRY = Registry("dataset", DatasetLoader())


@DATASET_REGISTRY.register()
def load_thumbnails(run: wandb_sdk.wandb_run.Run, config: DictConfig):
    artifact = run.use_artifact(f"{PROJECT_NAME}/{config.artifact_name}:latest", type="dataset")
    artifact_dir = artifact.download()
    df = pd.read_parquet(f"{artifact_dir}/{config.artifact_name}")
    df["path"] = df[config.column_name]
    if config.get("only_tma", False):
        df = df.query("is_tma")
    if config.get("only_thumbnail", False):
        df = df.query("thumbnail_path !=  image_path")
    train_df = df.query(f"fold != {config.fold}").reset_index(drop=True)
    val_df = df.query(f"fold == {config.fold}").reset_index(drop=True)
    if config.get("balance", False):
        max_images_per_label = train_df.groupby("label")["image_id"].count().max()
        train_df = (
            train_df.groupby("label")
            .sample(n=max_images_per_label, replace=True, random_state=config.seed)
            .reset_index(drop=True)
        )
    return train_df, val_df


@DATASET_REGISTRY.register()
def load_crop(run: wandb_sdk.wandb_run.Run, config: DictConfig):
    artifact = run.use_artifact(f"{PROJECT_NAME}/{config.artifact_name}:latest", type="dataset")
    artifact_dir = artifact.download()
    df = pd.read_parquet(f"{artifact_dir}/{config.artifact_name}")
    df["path"] = df[config.column_name]
    train_df = df.query(f"fold != {config.fold}").reset_index(drop=True)
    val_df = df.query(f"fold == {config.fold}").reset_index(drop=True)
    if config.get("balance", False):
        max_images_per_label = train_df.groupby("label")["image_id"].count().max()
        train_df = (
            train_df.groupby("label")
            .sample(n=max_images_per_label, replace=True, random_state=config.seed)
            .reset_index(drop=True)
        )
    return train_df, val_df


@DATASET_REGISTRY.register()
def load_tile(run: wandb_sdk.wandb_run.Run, config: DictConfig):
    artifact = run.use_artifact(f"{PROJECT_NAME}/{config.artifact_name}:latest", type="dataset")
    artifact_dir = artifact.download()
    df = pd.read_parquet(f"{artifact_dir}/{config.artifact_name}")
    df["path"] = df[config.column_name]
    if config.get("remove_edges", False):
        max_ij = df.groupby("image_id")[["i", "j"]].transform("max")
        df["max_i"] = max_ij["i"]
        df["max_j"] = max_ij["j"]
        df = df.query("i != max_i and j != max_j").reset_index(drop=True)
    if weight := config.get("weight", False):
        df = df.query(f"weight > {weight}")
    train_df = df.query(f"fold != {config.fold}").reset_index(drop=True)
    val_df = df.query(f"fold == {config.fold}").reset_index(drop=True)
    if mask_weight := config.get("mask_weight", False):
        train_df["mask_weight"] = train_df[label2idxmask.keys()].fillna(0).sum(1)
        train_df = train_df.query(f"mask_weight > {mask_weight}")
    if config.get("balance", False):
        max_images_per_label = train_df.groupby("label")["image_id"].count().max()
        train_df = (
            train_df.groupby("label")
            .sample(n=max_images_per_label, replace=True, random_state=config.seed)
            .reset_index(drop=True)
        )
    return train_df, val_df


@DATASET_REGISTRY.register()
def load_tile_model(run: wandb_sdk.wandb_run.Run, config: DictConfig):
    artifact = run.use_artifact(f"{PROJECT_NAME}/{config.artifact_name}:latest", type="dataset")
    artifact_dir = artifact.download()
    df = pd.read_parquet(f"{artifact_dir}/{config.artifact_name}")
    df["path"] = df[config.column_name]
    df['i'] = df['i'] // config.tile_size
    df['j'] = df['j'] // config.tile_size
    df['max_i'] = df.groupby(['image_id', 'component'])['i'].transform('max')
    df['max_j'] = df.groupby(['image_id', 'component'])['j'].transform('max')
    train_df = df.query(f"fold != {config.fold}").reset_index(drop=True)
    val_df = df.query(f"fold == {config.fold}").reset_index(drop=True)
    return train_df, val_df

