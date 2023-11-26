from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig
from ubc import get_configs, train_pl_run

ROOT_DIR = Path("../input/UBC-OCEAN/")

@hydra.main(config_path="ubc/configs", config_name="tf_efficientnet_b4_ns", version_base=None)
def train_pipeline(config: DictConfig) -> None:
    checkpoint_id = None
    img_sizes = [512, 1024, 1536 ,2048, 2048]
    learning_rates = [1e-4, 5e-5, 2e-5, 5e-6, 1e-5]
    batch_sizes = [32, 8, 4, 2, 2]
    datasets = ["thumbnail", "thumbnail", "thumbnail", "thumbnail", "crop-0.0-2048"]
    for img_size, lr, dataset, batch_size in zip(img_sizes, learning_rates, datasets, batch_sizes):
        config.img_size = img_size
        config.lr = lr
        config.batch_size = batch_size
        config.dataset = get_configs(folder="dataset")[dataset]
        config.checkpoint_id = checkpoint_id
        checkpoint_id = train_pl_run(config)
        logger.info(f"Checkpoint ID: {checkpoint_id}")
        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    train_pipeline() # pylint: disable=no-value-for-parameter
    print("Done!")
