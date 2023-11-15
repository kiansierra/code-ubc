
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

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsize',   type=int,   default=2048)
    parser.add_argument('--output-folder',   type=str,   default="output file path")
    parser.add_argument('--dataframe-path',   type=str,   default="Dataframe path")
    parser.add_argument('--num_processes',   type=int,   default=4)
    args = parser.parse_args()
    return args

def resize(image_path, output_path, imgsize=2048):
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
    df = pd.read_parquet(args.dataframe_path)
    df['output_path'] = df['image_id'].apply(lambda x: f"{args.output_folder}/{x}.png")
    records = df.to_dict('records')
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    resize_partial = partial(resize_copy, image_size=args.imgsize)
    with mp.Pool(args.num_processes) as pool:
        pool.map(resize_partial, records)
    df.drop(columns=['path'], inplace=True)
    df.rename(columns={'output_path': 'path'}, inplace=True)
    df.to_parquet(f"{args.output_folder}/train.parquet")
    
if __name__ == '__main__':
    main()
