
import argparse
import gc
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pyvips
from loguru import logger
from ubc import PyVipsProcessor, TorchProcessor

ROOT_DIR = Path("../input/UBC-OCEAN/")
PROCESSED_DIR = Path("../input/UBC-OCEAN-PROCESSED/")

    
def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='pyvips')
    parser.add_argument('--save_folder', type=str, default=str(PROCESSED_DIR / 'train'))
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--image_folder', type=str, default=None)
    parser.add_argument('--resize', type=float, default=0.25)
    parser.add_argument('--split_size', type=int, default=512)
    parser.add_argument('--max_tiles', type=int, default=None)
    parser.add_argument('--center_crop', action='store_true', default = False)
    parser.add_argument('--num_processes', type=int, default = 6)
    
    args = parser.parse_args()
    return args

def main() -> None:
    args = arguments()
    save_folder = Path(args.save_folder)
    assert args.image_path or args.image_folder, "Must provide either image_path or image_folder"
    assert args.backend in ['pyvips', 'torch'], "Backend must be one of ['pyvips', 'torch']"
    if args.backend == 'torch':
        processor = TorchProcessor(save_folder, args.resize, args.split_size,
                                max_tiles=args.max_tiles, center_crop=args.center_crop)
    else:
        
        processor = PyVipsProcessor(save_folder, args.resize, args.split_size,
                                max_tiles=args.max_tiles, center_crop=args.center_crop)
    if args.image_path:
        processor.run(args.image_path)
        return 
    all_images = list(Path(args.image_folder).glob('*.png'))
    with mp.Pool(processes=args.num_processes) as pool:
        pool.map(processor.run, all_images)
    
    
    
if __name__ == '__main__':
    main()




