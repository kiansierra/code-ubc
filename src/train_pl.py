import hydra
from omegaconf import DictConfig

from ubc import train_pl_run


@hydra.main(config_path="ubc/configs", config_name="tf_efficientnet_b4_ns", version_base=None)
def train(config: DictConfig) -> None:
    train_pl_run(config)


if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter
    print("Done!")
