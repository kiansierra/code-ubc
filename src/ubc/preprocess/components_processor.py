import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import cv2
import numpy as np
from PIL import Image

PositionalTuple = tuple[int, int]
ImageDict = Dict[PositionalTuple, Image.Image]
ArrayDict = Dict[PositionalTuple, np.ndarray]
FilterTypes = Literal['white', 'black']

def extract_components(img:np.ndarray, area_threshold:int=1_000, aspect_ratio_threshold:float=1.5):
    as_ratio = img.size[0] / img.size[1]
    if as_ratio <= aspect_ratio_threshold:
        return [(0, img.size[0]-1, 0,img.size[1]-1)]
    img = np.array(img)
    mask = np.max( img > 0, axis=-1 ).astype(np.uint8)
    retval, labels = cv2.connectedComponents(mask)
    positions = []
    for num in range(retval):
        component = ((labels ==num)[:,:,None] * img).max(2)
        area = (component > 0).sum()
        if area < area_threshold:
            continue
        x, y = np.where(component > 0)
        x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
        positions.append((x_min/img.shape[0], x_max/img.shape[0], y_min/img.shape[1], y_max/img.shape[1]))
    
    return positions

@dataclass
class ComponentsProcessor:
    save_folder:str
    resize_factor:float = 0.5
    split_size:int = 2048
    min_resize:int = 4000
    extension:str = 'png'
    empty_threshold:float = 0.5
    max_tiles:Optional[int] = None
    center_crop:bool = True
    filter_type:FilterTypes = 'black'
    
    def load_image(self, path:str):
        return  Image.open(path)
        
    def resize(self, img: Image) -> Image:
        if self.resize_factor != 1 and max(img.width, img.height) >self.min_resize: #Exculde TMAs which are already small
            img = img.resize(self.resize_factor)
        return img
    
    def crop(self, img: Image) -> Image:
        split_size = self.split_size
        if self.center_crop:
            new_width = img.width - img.width%split_size
            new_height = img.height - img.height%split_size
            img = img.crop((img.width - new_width)//2, (img.height - new_height)//2, new_width, new_height)
        return img
        
    def split_image(self, img: Image) -> ImageDict:
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
            output[(i,j)] = Image.fromarray(tile)
            if self.max_tiles and len(output) >= self.max_tiles:
                break
        return output
        
    def filter_tile(self, tile: np.ndarray) -> bool:
        if self.filter_type == 'black':
            keep = (tile>0).mean()>self.empty_threshold
        elif self.filter_type == 'white':
            keep = (tile >= 255).mean()<self.empty_threshold
        return keep
    
    def save_tile(self, tile: np.ndarray | Image.Image, path:Path):
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

            