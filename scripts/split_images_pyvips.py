
import argparse
import gc
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pyvips
from loguru import logger

ROOT_DIR = Path("../input/UBC-OCEAN/")
PROCESSED_DIR = Path("../input/UBC-OCEAN-PROCESSED/")


def split_image(img:np.ndarray, split_size:int=512) -> Dict[Tuple[int, int], pyvips.Image]:
    output = {}
    for i in range(0, img.width, split_size):
        for j in range(0, img.height, split_size):
            output[(i,j)] = img.crop(i, j, min(split_size, img.width-i), min(split_size, img.height-j))
    return output


def save_image_splits(img:np.ndarray, path:Path, split_size:int=512) -> None:
    splits = split_image(img, split_size)
    clean_emptys = lambda x: {k:v for k,v in x.items() if np.array(v).std()>0}
    splits = clean_emptys(splits)
    logger.info(f'Splitting image {path.stem} in to {len(splits)} splits of size {split_size}')
    for (i,j), split in splits.items():
        split.write_to_file(str(path / f'{i}_{j}.png'))


def load_split_save(img_path:Path, save_folder:Path, split_size:int=512, resize_factor:float=0.25) -> None:
    save_path = save_folder/ img_path.stem
    save_path.mkdir(exist_ok=True, parents=True)
    if len(list(save_path.glob('*.png'))) > 4:
        logger.info(f'Image {img_path.stem} already split. Skipping.')
        return None
    for img_path in save_path.glob("*.png"):
        os.remove(img_path)
    img = pyvips.Image.new_from_file(img_path)
    if resize_factor != 1 and max(img.width, img.height) >4_000: #Exculde TMAs which are already small
        img = img.resize(resize_factor)
        gc.collect()
    
    save_image_splits(img, save_path, split_size)
    del img
    gc.collect()
    
def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', type=str, default=str(PROCESSED_DIR / 'train'))
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--resize', type=float, default=0.25)
    parser.add_argument('--split_size', type=int, default=512)
    args = parser.parse_args()
    return args

def main() -> None:
    args = arguments()
    image_path = Path(args.image_path)
    save_folder = Path(args.save_folder)
    load_split_save(image_path, save_folder, args.split_size, args.resize)
    
if __name__ == '__main__':
    main()




