import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path("../input/UBC-OCEAN/")

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe-path', type=str, default=None, required=True)
    parser.add_argument('--image-folder', type=str, default=None, required=True)
    parser.add_argument('--save-folder', type=str, default=None, required=True)
    return parser.parse_args()


def main():
    args = parser()
    df = pd.read_parquet(args.dataframe_path)
    df.drop(columns=['path', 'label-tma'], inplace=True)
    image_dir = Path(args.image_folder)
    crops_df = pd.DataFrame({'path' :list(image_dir.rglob("*.png"))})
    crops_df['image_id'] = crops_df['path'].apply(lambda x: x.parent.stem).astype(int)
    crops_df['path'] = crops_df['path'].apply(lambda x: str(x))
    crops_df = crops_df.merge(df, on='image_id', how='left').reset_index(drop=True)
    crops_df.to_parquet(f'{args.save_folder}/train-{args.folder_name}.parquet')
    
if __name__ == '__main__':
    main()