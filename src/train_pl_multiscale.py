import copy
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig

from ubc import get_configs, train_pl_run

ROOT_DIR = Path("../input/UBC-OCEAN/")


@hydra.main(config_path="ubc/configs/multiscale", config_name="multiscale-base", version_base=None)
def train_pipeline(config: DictConfig) -> None:
    base_config = get_configs()[config.get("base_model")]
    config_lengths = {k: len(v) for k, v in config.items() if k != "base_model"}
    output_dir = copy.deepcopy(base_config.output_dir)
    assert len(set(config_lengths.values())) == 1, f"Lengths of config lists don't match {config_lengths}"
    checkpoint_id = None
    num_steps = list(config_lengths.values())[0]
    config_keys = ["dataset", "optimizer"]
    for num in range(num_steps):
        for k in config_lengths.keys():
            if k in config_keys:
                base_config.dataset = get_configs(folder=k)[config[k][num]]
            else:
                base_config[k] = config[k][num]
        base_config.checkpoint_id = checkpoint_id
        checkpoint_id = train_pl_run(base_config)
        base_config.output_dir = output_dir
        logger.info(f"Checkpoint ID: {checkpoint_id}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train_pipeline()  # pylint: disable=no-value-for-parameter
    print("Done!")
