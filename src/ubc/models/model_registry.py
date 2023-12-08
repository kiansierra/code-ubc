import pytorch_lightning as pl
from ..utils import Registry
from .optimization_utils import get_optimizer, get_scheduler
from omegaconf import DictConfig
from typing import Any


class ConfigLightningModel(pl.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

    def configure_optimizers(self) -> Any:
        optimizer = get_optimizer(self.config, self)
        scheduler = get_scheduler(self.config, optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


MODEL_REGISTRY = Registry("MODELS", ConfigLightningModel)
