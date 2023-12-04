import argparse
import multiprocessing as mp
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from ubc import PROJECT_NAME, tile_func, upload_to_wandb

import wandb

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None



def tile_wrapper(row:Dict[str, Any], tile_size:int, output_folder:str, empty_threshold:float, center_crop:bool=False):
    image_output_folder = f"{output_folder}/images"
    mask_output_folder = f"{output_folder}/masks"
    img = Image.open(row['path'])
    image_df = tile_func(img, row['image_id'], row['component'], tile_size, empty_threshold,
                         image_output_folder, center_crop=center_crop)
    if os.path.exists(row['mask_path']):
        mask = Image.open(row['mask_path'])
        mask_df = tile_func(mask, row['image_id'], row['component'], tile_size, 0.0,
                            mask_output_folder, get_weights=True, center_crop=center_crop)
        mask_df.rename(columns={'path': 'mask_path'}, inplace=True)
        image_df = image_df.merge(mask_df, on=['image_id', 'component', 'i', 'j'], how='left')
    return image_df

@hydra.main(config_path="../src/ubc/configs/preprocess", config_name="tile", version_base=None)
def main(config: DictConfig) -> None:
    if config.artifact_name:
        config_container = OmegaConf.to_container(config, resolve=True)
        run = wandb.init(job_type='tile', config=config_container, project=PROJECT_NAME)
        artifact = run.use_artifact(f"{config.artifact_name}:latest", type='dataset')
        artifact_dir = artifact.download()
        config.dataframe_path = f"{artifact_dir}/{config.artifact_name}"
    df = pd.read_parquet(config.dataframe_path)
    records = df.to_dict('records')
    output_folder = Path(config.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    tile_save = partial(tile_wrapper, output_folder=output_folder, tile_size=config.tile_size,
                        empty_threshold=config.empty_threshold, center_crop=config.center_crop)
    if config.num_processes > 1:
        with mp.Pool(config.num_processes) as pool:
            outputs = pool.map(tile_save, records)
    else:
        outputs = list(map(tile_save, records))
    output_df = pd.concat(outputs)
    output_df.reset_index(drop=True, inplace=True)
    df.rename(columns={'path': 'component_path', 'mask_path': 'mask_component_path'}, inplace=True)
    output_df = output_df.merge(df, on=['image_id', 'component'], how='left')
    output_df.to_parquet(f"{config.output_folder}/{config.output_name}")
    if config.artifact_name:
        artifact = upload_to_wandb(config.output_folder, config.output_name, pattern="*.parquet",
                                   artifact_type='dataset', end_wandb_run=False, return_artifact=True)
        table_name = config.output_name.replace('.parquet', '')
        artifact.add(wandb.Table(dataframe=output_df), name=table_name)
        run.log_artifact(artifact)
        run.finish()
    
if __name__ == '__main__':
    main()