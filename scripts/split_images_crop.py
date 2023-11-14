
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from ubc import ComponentsProcessor

load_dotenv()

ROOT_DIR = Path(os.environ.get("ROOT_DIR","../input/UBC-OCEAN/"))


def apply_processor(processor:ComponentsProcessor, record):
    processor.run(**record)

@hydra.main(config_path="../src/ubc/configs/preprocess", config_name="crop-512-0.5", version_base=None)
def main(config:DictConfig) -> None:
    save_folder = Path(config.save_folder)
    assert config.get('image_folder'), "Must provide either image_folder"
    logger.info(f"Loading {config.dataframe_path}")
    df = pd.read_parquet(config.dataframe_path)
    df_group = df.groupby('image_id', as_index=False)['components'].agg(list).rename(columns={'components': 'crop_position'})
    logger.info(f"Using Images in  {config.image_folder}")
    df_group['path'] = df_group['image_id'].apply(lambda x: f"{config.image_folder}/{x}.png")
    records = df_group[['path', 'crop_position']].to_dict("records")
    processor = ComponentsProcessor(save_folder, config.resize_factor, config.split_size,
                            max_tiles=config.max_tiles, center_crop=False)
    start_time = time.time()
    partial_apply_processor = partial(apply_processor, processor)
    logger.info(f"total files {len(df_group['path'].unique())} started at {time.ctime()}")
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




