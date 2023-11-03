from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = Path("../input/UBC-OCEAN/")

def get_path(row):
    if row["is_tma"]:
        return str(ROOT_DIR / "train_images" / f"{row['image_id']}.png")
    return str(ROOT_DIR / "train_thumbnails" / f"{row['image_id']}_thumbnail.png")

def generate_folds() -> None:
    train_df = pd.read_csv(ROOT_DIR / 'train.csv')
    train_df['path'] = train_df.apply(get_path, axis=1).astype(str)
    train_df['label-tma'] = train_df['label'].astype(str) + '-' + train_df['is_tma'].astype(str)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_df['fold'] = -1
    for fold, (train_index, valid_index) in enumerate(kfold.split(train_df, train_df['label-tma'])):
        train_df.loc[valid_index, 'fold'] = fold
    train_df.to_parquet(ROOT_DIR / 'train.parquet')

if __name__ == '__main__':
    generate_folds()



