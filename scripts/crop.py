import argparse
import multiprocessing as mp
import os
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

import wandb
from ubc import get_cropped_images, upload_to_wandb

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None


def crop(row, output_folder):
    image = Image.open(row['path'])
    data, images =  get_cropped_images(image, row['image_id'], output_folder)
    for path, img in zip(data['path'], images):
        img.save(path)
    return data


@hydra.main(config_path="../src/ubc/configs/preprocess", config_name="crop", version_base=None)
def main(config: DictConfig) -> None:
    if config.artifact_name:
        config_container = OmegaConf.to_container(config, resolve=True)
        run = wandb.init(job_type='crop', config=config_container)
        artifact = run.use_artifact(f"{config.artifact_name}:latest", type='dataset')
        artifact_dir = artifact.download()
        config.dataframe_path = f"{artifact_dir}/{config.artifact_name}"
    df = pd.read_parquet(config.dataframe_path)
    records = df.to_dict('records')
    output_folder = Path(config.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    crop_save = partial(crop, output_folder=output_folder)
    if config.num_processes > 1:
        with mp.Pool(config.num_processes) as pool:
            outputs = pool.map(crop_save, records)
    else:
        outputs = list(map(crop_save, records))
    output_df = pd.concat(outputs)
    output_df['component'] = output_df.index
    output_df.reset_index(drop=True, inplace=True)
    output_df = output_df.merge(df.rename(columns={'path': 'crop_path'}), on='image_id', how='left')
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