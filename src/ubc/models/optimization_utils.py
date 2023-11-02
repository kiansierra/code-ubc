from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..utils import get_builder


def get_optimizer(config:DictConfig, model:nn.Module) -> Optimizer:
    optimizer_bulder = get_builder(**config.optimizer)
    if hasattr(model, 'parameter_groups'):
        return optimizer_bulder(model.parameter_group())
    return optimizer_bulder(model.parameters())

def get_scheduler(config:DictConfig, optimizer:Optimizer) -> _LRScheduler:
    scheduler_builder = get_builder(**config.scheduler)
    return scheduler_builder(optimizer=optimizer)