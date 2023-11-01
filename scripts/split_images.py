
import argparse
import concurrent.futures
import gc
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Union

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import mlcrate as mlc
import numpy as np
import pandas as pd
import timm
from loguru import logger

ROOT_DIR = Path("../input/UBC-OCEAN/")
PROCESSED_DIR = Path("../input/UBC-OCEAN-PROCESSED/")





def split_image(img:np.ndarray, split_size:int=512) -> Dict[Tuple[int, int], np.ndarray]:
    output = {}
    for i in range(0, img.shape[0], split_size):
        for j in range(0, img.shape[1], split_size):
            output[(i,j)] = img[i:i+split_size, j:j+split_size]
    return output


def save_image_splits(img:np.ndarray, path:Path, split_size:int=512) -> None:
    splits = split_image(img, split_size)
    clean_emptys = lambda x: {k:v for k,v in x.items() if v.std()>0}
    splits = clean_emptys(splits)
    logger.info(f'Splitting image {path.stem} in to {len(splits)} splits of size {split_size}')
    for (i,j), split in splits.items():
        plt.imsave(str(path / f'{i}_{j}.png'), split)


def load_split_save(path:Path, split_size:int=512, resize_factor:float=0.25) -> None:
    save_path = PROCESSED_DIR / 'train' / path.stem
    save_path.mkdir(exist_ok=True, parents=True)
    if len(list(save_path.glob('*.png'))) > 0:
        logger.info(f'Image {path.stem} already split. Skipping.')
        return None
    img = plt.imread(str(path))
    if resize_factor != 1:
        max_size = int(max(img.shape)*resize_factor) 
        logger.info(f'Resizing image {path.stem} of shape {img.shape} by factor {resize_factor} to max size {max_size}')
        img = A.LongestMaxSize(max_size=max_size)(image=img)['image']
        gc.collect()
    save_path = PROCESSED_DIR / 'train' / path.stem
    save_path.mkdir(exist_ok=True, parents=True)
    
    save_image_splits(img, save_path, split_size)
    del img
    gc.collect()
    
def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--resize', type=float, default=0.25)
    parser.add_argument('--split_size', type=int, default=512)
    args = parser.parse_args()
    return args

def main() -> None:
    args = arguments()
    path = Path(args.image_path)
    load_split_save(path, args.split_size, args.resize)
    
if __name__ == '__main__':
    main()




