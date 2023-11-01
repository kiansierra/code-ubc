from pathlib import Path

import albumentations as A
import hydra
import pandas as pd
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from ubc import AugmentationDataset, TimmModel

ROOT_DIR = Path("../input/UBC-OCEAN/")

def get_train_transforms(config):
    return A.Compose([
        A.Resize(config['img_size'], config['img_size']),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)

@hydra.main(config_path='ubc/configs', config_name='config', version_base=None)
def train(config: DictConfig) -> None:
    df = pd.read_parquet(ROOT_DIR / 'train.parquet')
    df['path'] = df.apply(lambda x: str(ROOT_DIR/ 'train_images' / f"{x['image_id']}.png") if x['is_tma'] else str(ROOT_DIR/ 'train_thumbnails' / f"{x['image_id']}_thumbnail.png"), axis=1)
    
    train_df = df[df['fold'] != config['fold']].reset_index(drop=True)
    valid_df = df[df['fold'] == config['fold']].reset_index(drop=True)
    train_ds = AugmentationDataset(train_df, augmentation=get_train_transforms(config))
    valid_ds = AugmentationDataset(valid_df, augmentation=get_train_transforms(config))
    train_dataloader = DataLoader(train_ds, **config.dataloader.tr_dataloader)
    valid_dataloader = DataLoader(valid_ds, **config.dataloader.val_dataloader)
    model = TimmModel(config)
    trainer = pl.Trainer(**config['trainer'])
    trainer.fit(model, train_dataloader, valid_dataloader)
    
    
if __name__ == '__main__':
    train()
    
    