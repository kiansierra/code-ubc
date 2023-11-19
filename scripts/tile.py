import argparse
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path
from typing import Dict, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf
import cv2
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image
import warnings
import wandb
from ubc import upload_to_wandb

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

def tiler(img: Image.Image, tile_size: int) -> Dict[Tuple[int,int],Image.Image]:
    """Split image into tiles of size tile_size x tile_size with overlap
    Args:
        img (Image.Image): PIL image
        tile_size (int): tile size
    Returns:
        Dict[Image.Image]: list of tiles
    """
    img_array = np.array(img)
    x_splits = list(range(tile_size,img_array.shape[0],tile_size))
    y_splits = list(range(tile_size,img_array.shape[1],tile_size))
    all_splits = [np.split(elem, y_splits, 1) for elem in np.split(img_array, x_splits, 0)]
    output = {}
    for i, elem in enumerate(all_splits):
        for j, elem2 in enumerate(elem):
            output[(i*tile_size,j*tile_size)] = elem2
    return output

def tile_func(img_path, img_id, component, tile_size, output_folder):
    img = Image.open(img_path)
    logger.debug(f"tiling {img_id} with {img.size} and {tile_size=}")
    tiles = tiler(img, tile_size)
    data = []
    os.makedirs(f"{output_folder}/{img_id}", exist_ok=True)
    for (i,j), tile in tiles.items():
        tile = Image.fromarray(tile)
        path = f"{output_folder}/{img_id}/{component}_{i}_{j}.png"
        tile.save(path)
        data.append({'image_id': img_id, 'path':path, 'component': component, 'i': i, 'j': j})
    return pd.DataFrame(data)

def tile_wrapper(row, tile_size, output_folder):
    return tile_func(row['path'], row['image_id'], row['component'], tile_size, output_folder)

@hydra.main(config_path="../src/ubc/configs/preprocess", config_name="tile", version_base=None)
def main(config: DictConfig) -> None:
    if config.artifact_name:
        config_container = OmegaConf.to_container(config, resolve=True)
        run = wandb.init(job_type='tile', config=config_container)
        artifact = run.use_artifact(f"{config.artifact_name}:latest", type='dataset')
        artifact_dir = artifact.download()
        config.dataframe_path = f"{artifact_dir}/{config.artifact_name}"
    df = pd.read_parquet(config.dataframe_path)
    records = df.to_dict('records')
    output_folder = Path(config.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    tile_save = partial(tile_wrapper, output_folder=output_folder, tile_size=config.tile_size)
    if config.num_processes > 1:
        with mp.Pool(config.num_processes) as pool:
            outputs = pool.map(tile_save, records)
    else:
        outputs = list(map(tile_save, records))
    output_df = pd.concat(outputs)
    output_df.reset_index(drop=True, inplace=True)
    output_df = output_df.merge(df.rename(columns={'path': 'component_path'}), on=['image_id', 'component'], how='left')
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