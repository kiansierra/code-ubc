import inspect
import os
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn

__all__ = [
    "set_seed",
    "rebuild_layer",
    "freeze_layer",
    "unfreeze_layer",
    "init_layer",
    "freeze_batch_norms",
    "init_bn",
    "get_layer_signature",
]


def get_layer_signature(layer: nn.Module) -> Dict[str, Any]:
    named_inputs = [arg for arg in inspect.getfullargspec(type(layer)).args if arg != "self"]
    input_kwargs = {k: v for k, v in layer.__dict__.items() if k in named_inputs}
    if hasattr(layer, "bias"):
        input_kwargs["bias"] = layer.bias is not None
    return input_kwargs


def set_seed(seed: int = 42) -> None:
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def rebuild_layer(layer: nn.Module, keys: List[str], **kwargs) -> nn.Module:
    builder = type(layer)
    build_kwargs = {key: getattr(layer, key) for key in keys}
    for key, val in kwargs.items():
        if key in keys:
            build_kwargs[key] = val
    return builder(**build_kwargs)


def freeze_layer(layer: nn.Module) -> None:
    for param in layer.parameters():
        param.requires_grad = False


def unfreeze_layer(layer: nn.Module) -> None:
    for param in layer.parameters():
        param.requires_grad = True


def freeze_batch_norms(model: nn.Module) -> None:
    batch_norm_parent_class = nn.BatchNorm1d.__bases__[0]
    for layer in model.modules():
        if isinstance(layer, batch_norm_parent_class):
            freeze_layer(layer)


def init_layer(layer: nn.Module) -> None:
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(batch_norm: nn.Module) -> None:
    batch_norm.bias.data.fill_(0.0)
    batch_norm.weight.data.fill_(1.0)
