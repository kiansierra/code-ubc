import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pyvips
from torch import Tensor
from torchvision.io.image import decode_png, read_file, write_png
from torchvision.transforms import CenterCrop, Resize

PositionalTuple = tuple[int, int]
ImageDict = Dict[PositionalTuple, pyvips.Image]
ArrayDict = Dict[PositionalTuple, Tensor]


@dataclass
class TorchProcessor:
    save_folder: str
    resize_factor: float = 0.5
    split_size: int = 2048
    min_resize: int = 4000
    extension: str = "png"
    empty_threshold: float = 0.5
    max_tiles: Optional[int] = None
    center_crop: bool = True
    device: str = "cuda"

    def load_image(self, path: str):
        data = read_file(path)
        img_tensor = decode_png(data).to(self.device)
        return img_tensor

    def resize(self, img: Tensor) -> Tensor:
        if self.resize_factor != 1 and max(img.shape) > self.min_resize:  # Exculde TMAs which are already small
            new_width = int(img.shape[1] * self.resize_factor)
            new_height = int(img.shape[2] * self.resize_factor)
            img = Resize((new_width, new_height))(img)
        return img

    def crop(self, img: Tensor) -> Tensor:
        split_size = self.split_size
        if self.center_crop:
            new_width = img.shape[1] - (img.shape[1] % split_size)
            new_height = img.shape[2] - (img.shape[2] % split_size)
            img = CenterCrop((new_width, new_height))(img.squeeze())
        return img

    def split_image(self, img: Tensor) -> ImageDict:
        split_size = self.split_size
        tiles = [elem.split(512, 2) for elem in img.split(512, 1)]
        output = {}
        for i, row in enumerate(tiles):
            for j, tile in enumerate(row):
                if not self.filter_tile(tile):
                    continue
                output[(i * split_size, j * split_size)] = tile
        idxs = list(output.keys())
        if self.max_tiles is not None:
            random.shuffle(idxs)
            output = {k: output[k] for k in idxs[: self.max_tiles]}
        return output

    def filter_tile(self, tile: Tensor) -> bool:
        keep = (tile > 0).half().mean() > self.empty_threshold
        return keep

    def save_tile(self, tile: Tensor, path: Path):
        if path.suffix == ".npy":
            np.save(path, tile.numpy())
        else:
            write_png(tile.cpu(), str(path))

    def run(self, path: str):
        img_path = Path(path)
        save_path = Path(self.save_folder) / img_path.stem
        save_path.mkdir(exist_ok=True, parents=True)
        img = self.load_image(path)
        img = self.resize(img)
        img = self.crop(img)
        chunks = self.split_image(img)
        for (i, j), chunk in chunks.items():
            self.save_tile(chunk, save_path / f"{i}_{j}.{self.extension}")
        return chunks
