from typing import Any, Dict, Optional, Union

import torch

DataElement = Dict[str, Any]
BatchElement = Dict[str, torch.Tensor]
ElementBatch = Dict[str, Any]
TensorBatch = Dict[str, torch.Tensor]
DeviceType = Union[str, torch.device]
OptTensor = Optional[torch.Tensor]

__all__ = ["DataElement", "BatchElement", "DeviceType", "OptTensor", "ElementBatch", "TensorBatch"]
