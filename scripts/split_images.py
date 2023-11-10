
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
from omegaconf import DictConfig

from ubc import PyVipsProcessor, TorchProcessor

ROOT_DIR = Path("../input/UBC-OCEAN/")
PROCESSED_DIR = Path("../input/UBC-OCEAN-PROCESSED/")

    

@hydra.main(config_path="../src/ubc/configs/preprocess", config_name="512-0.25", version_base=None)
def main(config:DictConfig) -> None:
    save_folder = Path(config.save_folder)
    assert config.get('image_path') or config.get('image_folder'), "Must provide either image_path or image_folder"
    assert config.backend in ['pyvips', 'torch'], "Backend must be one of ['pyvips', 'torch']"
    if config.backend == 'torch':
        processor = TorchProcessor(save_folder, config.resize_factor, config.split_size,
                                max_tiles=config.max_tiles, center_crop=config.center_crop)
    else:
        
        processor = PyVipsProcessor(save_folder, config.resize_factor, config.split_size,
                                max_tiles=config.max_tiles, center_crop=config.center_crop)
    if config.get('image_path'):
        processor.run(config.image_path)
        return 
    all_images = list(Path(config.image_folder).glob('*.png'))
    with mp.Pool(processes=config.num_processes) as pool:
        pool.map(processor.run, all_images)
    
    
    
if __name__ == '__main__':
    main()




