import argparse
from pathlib import Path

import pandas as pd

ROOT_DIR = Path("../input/UBC-OCEAN/")
PROCESSED_DIR = Path("../input/UBC-OCEAN-PROCESSED/")

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder-name', type=str, default=None, required=True)
    return parser.parse_args()


def main():
    args = parser()
    df = pd.read_parquet(ROOT_DIR / "train.parquet")
    df.drop(columns=['path', 'label-tma'], inplace=True)
    IMAGE_DIR = PROCESSED_DIR / args.folder_name
    crops_df = pd.DataFrame({'path' :list(IMAGE_DIR.rglob("*.png"))})
    crops_df['image_id'] = crops_df['path'].apply(lambda x: x.parent.stem).astype(int)
    crops_df['path'] = crops_df['path'].apply(lambda x: str(x))
    crops_df = crops_df.merge(df, on='image_id', how='left').reset_index(drop=True)
    crops_df.to_parquet(ROOT_DIR / f'train-{args.folder_name}.parquet')
    
if __name__ == '__main__':
    main()