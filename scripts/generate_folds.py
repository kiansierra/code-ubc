
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import mlcrate as mlc
import numpy as np
import pandas as pd
import timm
from PIL import Image
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = Path("../input/UBC-OCEAN/")


def generate_folds() -> None:
    train_df = pd.read_csv(ROOT_DIR / 'train.csv')
    train_df['label-tma'] = train_df['label'].astype(str) + '-' + train_df['is_tma'].astype(str)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_df['fold'] = -1
    for fold, (train_index, valid_index) in enumerate(kfold.split(train_df, train_df['label-tma'])):
        train_df.loc[valid_index, 'fold'] = fold
    train_df.to_parquet(ROOT_DIR / 'train.parquet')

if __name__ == '__main__':
    generate_folds()



