
import argparse
import multiprocessing as mp
import os
import shutil
import warnings
from functools import partial
from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from ubc import PROJECT_NAME, resize, upload_to_wandb

import wandb

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None


    
def resize_copy(row, output_folder:str,  scale:float, image_size:int=2048):
    output_path = f"{output_folder}/images/{row['image_id']}.png"
    mask_output_path = f"{output_folder}/masks/{row['image_id']}.png"
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
        
    image = Image.open( row['path'])
    image = resize(image, scale, image_size)
    image.save( output_path )
    logger.info(f"save done {output_path}")
    outputs = {'image_id': row['image_id'], 'path': output_path, 'mask_path': '', 'size': image.size}
    if  os.path.exists(row['mask_path']):
        mask = Image.open( row['mask_path'])
        mask = resize(mask, scale, image_size)
        mask.save( mask_output_path )
        logger.info(f"mask save done {mask_output_path}")
        outputs['mask_path'] = mask_output_path
    return outputs
    
@hydra.main(config_path="../src/ubc/configs/preprocess", config_name="resize", version_base=None)
def main(config: DictConfig):
    if config.artifact_name:
        config_container = OmegaConf.to_container(config, resolve=True)
        run = wandb.init(job_type='resize', config=config_container, project=PROJECT_NAME)
        artifact = run.use_artifact(f"{config.artifact_name}:latest", type='dataset')
        artifact_dir = artifact.download()
        config.dataframe_path = f"{artifact_dir}/{config.artifact_name}"
    df = pd.read_parquet(config.dataframe_path)
    df['path'] = df[config.column_name]
    records = df.to_dict('records')
    output_folder = Path(config.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    resize_partial = partial(resize_copy, output_folder=config.output_folder,
                             image_size=config.imgsize, scale=config.scale)
    if config.num_processes <= 1:
        outputs = list(map(resize_partial, records))
    else:
        with mp.Pool(config.num_processes) as pool:
            outputs = pool.map(resize_partial, records)
    outputs_df = pd.DataFrame(outputs)
    df = outputs_df.merge(df[['image_id', 'is_tma', 'label', 'fold', "image_path","thumbnail_path"]], on='image_id', how='left')
    df.to_parquet(f"{config.output_folder}/{config.output_name}")
    if config.artifact_name:
        artifact = upload_to_wandb(config.output_folder, config.output_name, pattern="*.parquet",
                                   artifact_type='dataset', end_wandb_run=False, return_artifact=True)
        table_name = config.output_name.replace('.parquet', '')
        artifact.add(wandb.Table(dataframe=df), name=table_name)
        run.log_artifact(artifact)
        run.finish()
    
if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
