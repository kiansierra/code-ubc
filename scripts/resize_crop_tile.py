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
from ubc import get_cropped_images, get_crops_from_data, resize, tile_func, upload_to_wandb

import wandb

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None



def resize_crop_tile(group, scale:float, image_size:int=2048, tilesize:int=512, empty_treshold:float=0.1,
                     output_folder:str=None, center_crop:bool=False):
    img_path = group.iloc[0]['path']
    img_id = group.iloc[0]['image_id']
    logger.info(f"Starting processing for {img_id=}")
    image = Image.open(img_path)
    image = resize(image, scale, image_size)
    logger.info(f"resize done for {img_id=}")
    data, images = get_crops_from_data(image, group, output_folder)
    logger.info(f"crop done for {img_id=}")
    dfs = []
    for component_num, img in zip(group['component'], images):
        tile_df = tile_func(img, img_id, component_num, tilesize, empty_treshold, output_folder, center_crop=center_crop)
        dfs.append(tile_df)
        logger.info(f"tilling done for {img_id=} {component_num=} to {len(tile_df)} tiles")
    return pd.concat(dfs)

@hydra.main(config_path="../src/ubc/configs/preprocess", config_name="resize-crop-tile", version_base=None)
def main(config: DictConfig) -> None:
    if config.artifact_name:
        config_container = OmegaConf.to_container(config, resolve=True)
        run = wandb.init(job_type='resize_crop', config=config_container)
        artifact = run.use_artifact(f"{config.artifact_name}:latest", type='dataset')
        artifact_dir = artifact.download()
        config.dataframe_path = f"{artifact_dir}/{config.artifact_name}"
    df = pd.read_parquet(config.dataframe_path)
    df['path'] = df[config.column_name]
    groups = df.groupby('image_id')
    groups = [g_df for k, g_df in groups]
    
    output_folder = Path(config.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    resize_copy_partial = partial(resize_crop_tile, output_folder=config.output_folder,
                                  image_size=config.imgsize, scale=config.scale,
                                  empty_treshold=config.empty_threshold, tilesize=config.tilesize,
                                  center_crop=config.center_crop)
    if config.num_processes > 1:
        with mp.Pool(config.num_processes) as pool:
            outputs = pool.map(resize_copy_partial, groups)
    else:
        outputs = list(map(resize_copy_partial, groups))
    output_df = pd.concat(outputs, ignore_index=True)
    output_df = output_df.merge(df.rename(columns={'path': 'original_path'}), on=['image_id', 'component'], how='left')
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