
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from ubc import ComponentsProcessor

ROOT_DIR = Path("../input/UBC-OCEAN/")
PROCESSED_DIR = Path("../input/UBC-OCEAN-PROCESSED/")

def apply_processor(processor:ComponentsProcessor, record):
    processor.run(**record)

@hydra.main(config_path="../src/ubc/configs/preprocess", config_name="crop-512-0.5", version_base=None)
def main(config:DictConfig) -> None:
    save_folder = Path(config.save_folder)
    assert config.get('image_folder'), "Must provide either image_path or image_folder"
    df = pd.read_parquet(ROOT_DIR / "train-components.parquet")
    df_group = df.groupby('image_id', as_index=False)['components'].agg(list).rename(columns={'components': 'crop_position'})
    df_group['path'] = df_group['image_id'].apply(lambda x: str(ROOT_DIR/ 'train_images' / f"{x}.png"))
    records = df_group[['path', 'crop_position']].to_dict("records")
    processor = ComponentsProcessor(save_folder, config.resize_factor, config.split_size,
                            max_tiles=config.max_tiles, center_crop=False)
    start_time = time.time()
    partial_apply_processor = partial(apply_processor, processor)
    logger.info(f"Processing {config.image_folder} total files {len(df_group['path'].unique())} started at {time.ctime()}")
    if config.num_processes > 1:
        with mp.Pool(processes=config.num_processes) as pool:
            pool.map(partial_apply_processor, records)
        time_elapsed = time.time() - start_time
        logger.info(f"Finished processing {config.image_folder} in {time_elapsed/60:.2f} minutes")
        return
    list(map(partial_apply_processor, records))
    time_elapsed = time.time() - start_time
    logger.info(f"Finished processing {config.image_folder} in {time_elapsed/60:.2d} minutes")
    
    
    
    
if __name__ == '__main__':
    main()




