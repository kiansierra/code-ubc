import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import cv2
import numpy as np
from PIL import Image

PositionalTuple = tuple[int, int]
ImageDict = Dict[PositionalTuple, Image.Image]
ArrayDict = Dict[PositionalTuple, np.ndarray]
FilterTypes = Literal["white", "black"]
CropPositions = List[tuple[int, int, int, int]]
Loaders = Literal["cv2", "pil"]


def extract_components(img: np.ndarray, area_threshold: int = 1_000, aspect_ratio_threshold: float = 1.5):
    as_ratio = img.size[0] / img.size[1]
    if as_ratio <= aspect_ratio_threshold:
        return [(0, img.size[0] - 1, 0, img.size[1] - 1)]
    img = np.array(img)
    mask = np.max(img > 0, axis=-1).astype(np.uint8)
    retval, labels = cv2.connectedComponents(mask)
    positions = []
    for num in range(retval):
        component = ((labels == num)[:, :, None] * img).max(2)
        area = (component > 0).sum()
        if area < area_threshold:
            continue
        x, y = np.where(component > 0)
        x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
        positions.append((x_min / img.shape[0], x_max / img.shape[0], y_min / img.shape[1], y_max / img.shape[1]))

    return positions


@dataclass
class ComponentsProcessor:
    save_folder: str
    resize_factor: float = 0.5
    split_size: int = 2048
    min_resize: int = 4000
    extension: str = "png"
    empty_threshold: float = 0.5
    max_tiles: Optional[int] = None
    center_crop: bool = True
    filter_type: FilterTypes = "black"
    loader: Loaders = "cv2"

    def load_image(self, path: str):
        if self.loader == "cv2":
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        return Image.open(path)

    def resize(self, img: Image) -> Image:
        if self.resize_factor != 1 and max(img.shape) > self.min_resize:  # Exculde TMAs which are already small
            new_width, new_height = int(self.resize_factor * img.shape[0]), int(self.resize_factor * img.shape[1])
            img = cv2.resize(img, (new_height, new_width))
        return img

    def crop(self, img: np.ndarray, crop_positions: Optional[CropPositions] = None) -> Image:
        img_width, img_height = img.shape[0], img.shape[1]
        if CropPositions is not None:
            crops = []
            for crop_position in crop_positions:
                crop_position = (
                    int(crop_position[0] * img_width),
                    int(crop_position[1] * img_width),
                    int(crop_position[2] * img_height),
                    int(crop_position[3] * img_height),
                )
                crops.append(img[crop_position[0] : crop_position[1], crop_position[2] : crop_position[3]])
            return crops
        split_size = self.split_size
        if self.center_crop:
            spare_width = img_width % split_size // 2
            spare_height = img_height % split_size // 2
            new_width = img_width - img_width % split_size
            new_height = img_height - img_height % split_size
            img = img[spare_width : new_width + spare_width, spare_height : new_height + spare_height]
        return img

    def split_image(self, img: np.ndarray | List[np.ndarray]) -> ImageDict:
        output = {}
        img_width, img_height = img.shape[0], img.shape[1]
        split_size = self.split_size
        idxs = [(x, y) for x in range(0, img_width, split_size) for y in range(0, img_height, split_size)]
        if self.max_tiles is not None:
            random.shuffle(idxs)
        for i, j in idxs:
            tile = img[i : min(i + split_size, img_width), j : min(j + split_size, img_height)]
            if not self.filter_tile(tile):
                continue
            output[(i, j)] = tile
            if self.max_tiles and len(output) >= self.max_tiles:
                break
        return output

    def filter_tile(self, tile: np.ndarray) -> bool:
        if self.filter_type == "black":
            keep = (tile > 0).mean() > self.empty_threshold
        elif self.filter_type == "white":
            keep = (tile >= 255).mean() < self.empty_threshold
        return keep

    def save_tile(self, tile: np.ndarray | Image.Image, path: Path):
        if path.suffix == ".npy":
            np.save(path, tile)
        else:
            tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), tile)

    def run(self, path: str, crop_position: Optional[CropPositions] = None):
        img_path = Path(path)
        save_path = Path(self.save_folder) / img_path.stem
        save_path.mkdir(exist_ok=True, parents=True)
        img = self.load_image(path)
        img = self.resize(img)
        img = self.crop(img, crop_position)
        if isinstance(img, list):
            for num, img_crop in enumerate(img):
                chunks = self.split_image(img_crop)
                for (i, j), chunk in chunks.items():
                    self.save_tile(chunk, save_path / f"{num}_{i}_{j}.{self.extension}")
            return
        chunks = self.split_image(img)
        for (i, j), chunk in chunks.items():
            self.save_tile(chunk, save_path / f"{i}_{j}.{self.extension}")
