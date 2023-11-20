
import argparse
import multiprocessing as mp
import os
import shutil
import warnings
from functools import partial
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import cv2
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image
from ubc import upload_to_wandb, resize

import wandb

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None



# def resize(image_path, output_path, scale, imgsize=2048):
    # if os.path.exists(output_path):
    #     logger.info(f"skipping {output_path}")
    #     return
    # image = Image.open( image_path)
    # w, h = image.size

    # ratio = imgsize / min([w, h])
    # ratio = max(ratio, scale)
    # w2 = int(w*ratio)
    # h2 = int(h*ratio)

    # image = image.resize( (w2, h2) )
    # logger.debug(f"{output_path} debug: ({w}, {h}) --> ({w2}, {h2})")
    # if not os.path.exists(os.path.dirname(output_path)):
    #     os.makedirs(os.path.dirname(output_path))
    # image.save( output_path )
    # logger.info(f"save done {output_path}")
    
def resize_copy(row, scale:float, image_size:int=2048):
    output_path = row['output_path']
    if os.path.exists(output_path):
        logger.info(f"skipping {output_path}")
        return
    if row['is_tma']:
        shutil.copy(row['path'], output_path)
        return
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image = Image.open( row['path'])
    image = resize(image, scale, image_size)
    image.save( output_path )
    logger.info(f"save done {output_path}")
    
@hydra.main(config_path="../src/ubc/configs/preprocess", config_name="resize", version_base=None)
def main(config: DictConfig):
    if config.artifact_name:
        config_container = OmegaConf.to_container(config, resolve=True)

        run = wandb.init(job_type='resize', config=config_container)
        artifact = run.use_artifact(f"{config.artifact_name}:latest", type='dataset')
        artifact_dir = artifact.download()
        config.dataframe_path = f"{artifact_dir}/{config.artifact_name}"
    df = pd.read_parquet(config.dataframe_path)
    df['path'] = df[config.column_name]
    df['output_path'] = df['image_id'].apply(lambda x: f"{config.output_folder}/{x}.png")
    records = df.to_dict('records')
    output_folder = Path(config.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    resize_partial = partial(resize_copy, image_size=config.imgsize, scale=config.scale)
    if config.num_processes <= 1:
        list(map(resize_partial, records))
    else:
        with mp.Pool(config.num_processes) as pool:
            pool.map(resize_partial, records)
    df.drop(columns=['path'], inplace=True)
    df.rename(columns={'output_path': 'path'}, inplace=True)
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
