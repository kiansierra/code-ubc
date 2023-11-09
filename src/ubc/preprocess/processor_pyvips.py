import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pyvips
from PIL import Image

PositionalTuple = tuple[int, int]
ImageDict = Dict[PositionalTuple, pyvips.Image]
ArrayDict = Dict[PositionalTuple, np.ndarray]

@dataclass
class PyVipsProcessor:
    save_folder:str
    resize_factor:float = 0.5
    split_size:int = 2048
    min_resize:int = 4000
    extension:str = 'png'
    empty_threshold:float = 0.5
    max_tiles:Optional[int] = None
    center_crop:bool = True
    
    def load_image(self, path:str):
        return  pyvips.Image.new_from_file(path)
        
    def resize(self, img: pyvips.Image) -> pyvips.Image:
        if self.resize_factor != 1 and max(img.width, img.height) >self.min_resize: #Exculde TMAs which are already small
            img = img.resize(self.resize_factor)
        return img
    
    def crop(self, img: pyvips.Image) -> pyvips.Image:
        split_size = self.split_size
        if self.center_crop:
            new_width = img.width - img.width%split_size
            new_height = img.height - img.height%split_size
            img = img.crop((img.width - new_width)//2, (img.height - new_height)//2, new_width, new_height)
        return img
        
    def split_image(self, img: pyvips.Image) -> ImageDict:
        output = {}
        split_size = self.split_size
        idxs = [(x,y) for x in range(0, img.width, split_size) for y in range(0, img.height, split_size)]
        if self.max_tiles is not None:
            random.shuffle(idxs)
        for i, j in idxs:
            tile = img.crop(i, j, min(split_size, img.width-i), min(split_size, img.height-j))
            tile = tile.numpy()
            if not self.filter_tile(tile):
                continue
            # tile = self.process_image(tile)
            new_size = int(split_size * self.resize_factor), int(split_size * self.resize_factor)
            tile = Image.fromarray(tile).resize(new_size, Image.LANCZOS)
            output[(i,j)] = tile
            if self.max_tiles and len(output) >= self.max_tiles:
                break
        return output
        
    def filter_tile(self, tile: np.ndarray) -> bool:
        keep = (tile>0).mean()>self.empty_threshold
        return keep
    
    def save_tile(self, tile: np.ndarray | pyvips.Image, path:Path):
        if path.suffix == '.npy':
            np.save(path, tile)
        else:
            tile.save(path)
        
    def run(self, path:str):
        img_path = Path(path)
        save_path = Path(self.save_folder) / img_path.stem
        save_path.mkdir(exist_ok=True, parents=True)
        img = self.load_image(path)
        img = self.resize(img)
        img = self.crop(img)
        chunks = self.split_image(img)
        for (i,j), chunk in chunks.items():
            self.save_tile(chunk, save_path / f'{i}_{j}.{self.extension}')
        return chunks
            