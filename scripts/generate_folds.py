from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

import wandb
from ubc import upload_to_wandb
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = Path("../input/UBC-OCEAN/")
PROCESSED_DIR = Path("../input/UBC-OCEAN-PROCESSED/")


def get_thumbnail(row):
    if row["is_tma"]:
        return str(ROOT_DIR / "train_images" / f"{row['image_id']}.png")
    return str(ROOT_DIR / "train_thumbnails" / f"{row['image_id']}_thumbnail.png")

def get_path(row):
    return str(ROOT_DIR / "train_images" / f"{row['image_id']}.png")


def generate_folds() -> None:
    train_df = pd.read_csv(ROOT_DIR / 'train.csv')
    train_df['thumbnail_path'] = train_df.apply(get_thumbnail, axis=1).astype(str)
    train_df['image_path'] = train_df.apply(get_path, axis=1).astype(str)
    train_df['label-tma'] = train_df['label'].astype(str) + '-' + train_df['is_tma'].astype(str)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_df['fold'] = -1
    for fold, ( _, valid_index) in enumerate(kfold.split(train_df, train_df['label-tma'])):
        train_df.loc[valid_index, 'fold'] = fold
    train_df.drop(columns=['label-tma'], inplace=True)
    train_df.to_parquet(PROCESSED_DIR / 'train_folds.parquet')
    artifact = upload_to_wandb(PROCESSED_DIR, 'train_folds.parquet', pattern="*.parquet", artifact_type='dataset',
                               job_type='generate_folds', end_wandb_run=False, return_artifact=True)
    run = wandb.run
    table = wandb.Table(dataframe=train_df)
    artifact.add(table, 'train_folds')
    run.log_artifact(artifact)
    run.finish()
    
    

if __name__ == '__main__':
    generate_folds()



