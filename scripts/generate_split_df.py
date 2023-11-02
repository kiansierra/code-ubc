from pathlib import Path

import pandas as pd

ROOT_DIR = Path("../input/UBC-OCEAN/")
PROCESSED_DIR = Path("../input/UBC-OCEAN-PROCESSED/")

def main():
    df = pd.read_parquet(ROOT_DIR / "train.parquet")
    crops_df = pd.DataFrame({'path' :list(PROCESSED_DIR.rglob("*.png"))})
    crops_df['image_id'] = crops_df['path'].apply(lambda x: x.parent.stem).astype(int)
    crops_df['path'] = crops_df['path'].apply(lambda x: str(x))
    crops_df = crops_df.merge(df, on='image_id', how='left').reset_index(drop=True)
    crops_df.to_parquet(ROOT_DIR / 'train-crops.parquet')
    
if __name__ == '__main__':
    main()