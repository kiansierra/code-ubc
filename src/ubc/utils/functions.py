import functools
import importlib
import inspect
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from omegaconf import DictConfig, open_dict

# from .logger import get_logger

__all__ = ["filter_arguments", "get_builder", "compose", "set_debug", "set_overfit"]


def filter_arguments(
    function: Callable, drop_keys: Optional[List[str]] = None, **kwargs: Optional[Any]
) -> Dict[str, Any]:
    """Removes key word arguments that are not applicable to the function

    Args:
        function (Callable): function to filter arguments for
        drop_keys (Optional[List], optional): argument names to always drop. Defaults to None.

    Returns:
        Dict[str, Any]: dictionary with arguments applicable to the function
    """
    inspected = inspect.getfullargspec(function)
    if inspected.varkw:
        return {k: v for k, v in kwargs.items() if k not in (drop_keys or [])}
    return {k: v for k, v in kwargs.items() if k in inspected.args}


def get_builder(**kwargs: Optional[Any]) -> Callable:
    """Loads a function from a module and aplies the corresponding key word arguments

    Raises:
        ValueError: If named object can't be found in the corresponding module

    Returns:
        Callable: function with remaining arguments to instantiate
    """
    module_name = str(kwargs["module_name"])
    module = importlib.import_module(module_name)
    builder = getattr(module, kwargs["obj_name"], None)  # type: ignore[arg-type]
    if builder is None:
        raise ValueError(f"Invalid object name {kwargs['obj_name']}")
    filtered_kwargs = filter_arguments(builder, **kwargs)
    used_keys = [k for k in kwargs.keys() if k in filtered_kwargs] + ["module_name", "obj_name"]
    unused_keys = [k for k in kwargs.keys() if k not in used_keys]
    if unused_keys:
        logger.debug(f"Unused keys for {kwargs['obj_name']}: {unused_keys}")
    return partial(builder, **filtered_kwargs)  # type: ignore[return-value]


def compose(*functions) -> Callable:
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)


def set_debug(config: DictConfig) -> DictConfig:
    """Sets the config to debug mode
    Args:
        config (DictConfig): Config to set to debug mode
    Returns:
        DictConfig: Config in debug mode
    """
    os.environ["WANDB_MODE"] = "dryrun"
    with open_dict(config):
        config.num_workers = 0
        config.dataloader.persistent_workers = False
        config.batch_size = 2
        config.max_steps = 125
        config.checkpoint_dir = f"{config.output_dir}/debug/{config.artifact_name}"
        config.save_model = True
        config.gradient_accumulation_steps = 2
        config.trainer.logging_steps = 50
    return config


def set_overfit(config: DictConfig) -> DictConfig:
    """Sets the config to overfit mode
    Args:
        config (DictConfig): Config to set to overfit mode
    Returns:
        DictConfig: Config in overfit mode
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with open_dict(config):
        config.checkpoint_dir = f"{config.output_dir}/overfit/{config.artifact_name}"
        config.save_model = True
        config.max_epochs = 100
        config.trainer.logging_steps = 100
        config.save_model = False
    return config
