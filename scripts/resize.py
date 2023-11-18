
import argparse
import multiprocessing as mp
import os
import shutil
import warnings
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image
from ubc import upload_to_wandb

import wandb

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsize',   type=int,   default=2048)
    parser.add_argument('--output-folder',   type=str,   default="output file path")
    parser.add_argument('--output-name',   type=str,   default="train-resize.parquet")
    parser.add_argument('--column-name',   type=str,   default="thumbnail_path")
    parser.add_argument('--dataframe-path',   type=str,   default="Dataframe path")
    parser.add_argument('--artifact-name',   type=str,   default=None)
    parser.add_argument('--num_processes',   type=int,   default=4)
    args = parser.parse_args()
    return args

def resize(image_path, output_path, imgsize=2048):
    if os.path.exists(output_path):
        logger.info(f"skipping {output_path}")
        return
    image = Image.open( image_path)
    w, h = image.size

    ratio = imgsize / min([w, h])
    w2 = int(w*ratio)
    h2 = int(h*ratio)

    image = image.resize( (w2, h2) )
    logger.debug(f"{output_path} debug: ({w}, {h}) --> ({w2}, {h2})")
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    image.save( output_path )
    logger.info(f"save done {output_path}")
    
def resize_copy(row, image_size=2048):
    if row['is_tma']:
        shutil.copy(row['path'], row['output_path'])
    else:
        resize(row['path'], row['output_path'], image_size)
    
def main():
    args = args_parser()
    if args.artifact_name:
        run = wandb.init(job_type='resize', config=args)
        artifact = run.use_artifact(f"{args.artifact_name}:latest", type='dataset')
        artifact_dir = artifact.download()
        args.dataframe_path = f"{artifact_dir}/{args.artifact_name}"
    df = pd.read_parquet(args.dataframe_path)
    df['path'] = df[args.column_name]
    df['output_path'] = df['image_id'].apply(lambda x: f"{args.output_folder}/{x}.png")
    records = df.to_dict('records')
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    resize_partial = partial(resize_copy, image_size=args.imgsize)
    with mp.Pool(args.num_processes) as pool:
        pool.map(resize_partial, records)
    df.drop(columns=['path'], inplace=True)
    df.rename(columns={'output_path': 'path'}, inplace=True)
    df.to_parquet(f"{args.output_folder}/{args.output_name}")
    if args.artifact_name:
        artifact = upload_to_wandb(args.output_folder, args.output_name, pattern="*.parquet",
                                   artifact_type='dataset', end_wandb_run=False, return_artifact=True)
        table_name = args.output_name.replace('.parquet', '')
        artifact.add(wandb.Table(dataframe=df), name=table_name)
        run.log_artifact(artifact)
        run.finish()
    
if __name__ == '__main__':
    main()
