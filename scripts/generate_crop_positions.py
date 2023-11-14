import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv
from ubc import extract_components_positions

load_dotenv()

ROOT_DIR = Path(os.environ.get("ROOT_DIR","../input/UBC-OCEAN/"))

def get_crop_components(row :Dict[str, Any]):
    if row['aspect_ratio'] >= 1.5:
        return extract_components_positions(row['path'])
    else:
        return [(0, (row['image_width']-1)/row['image_width'], 0, (row['image_height']-1)/row['image_height'])]

def main():
    train_df = pd.read_parquet(ROOT_DIR / "train.parquet")
    train_df['aspect_ratio'] = train_df['image_width'] / train_df['image_height']
    train_df['components'] = train_df.apply(get_crop_components, axis=1)
    train_df = train_df.explode('components').sort_values(by='image_id').reset_index(drop=True)
    train_df['crop'] = train_df.groupby('image_id')['components'].rank()
    train_df['weight'] = train_df['components'].apply(lambda x: (x[1] - x[0]) * (x[3] - x[2]))
    train_df.to_parquet(ROOT_DIR / "train-components.parquet")
    
if __name__ == '__main__':
    main()