from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pyvips

DictKey = tuple[int, int]

@dataclass
class PyVipsProcessor:
    save_folder:str
    resize_factor:float = 0.25
    split_size:int = 512
    min_resize:int = 4_000
    
    def load_image(self, path:str):
        return  pyvips.Image.new_from_file(path)
        
    def process_image(self, img: pyvips.Image) -> pyvips.Image:
        if self.resize_factor != 1 and max(img.width, img.height) >self.min_resize: #Exculde TMAs which are already small
            img = img.resize(self.resize_factor)
        return img
        
    def split_image(self, img: pyvips.Image) -> Dict[DictKey,pyvips.Image]:
        output = {}
        split_size = self.split_size
        for i in range(0, img.width, split_size):
            for j in range(0, img.height, split_size):
                output[(i,j)] = img.crop(i, j, min(split_size, img.width-i), min(split_size, img.height-j))
        return output
        
    def filter_chunks(self, chunk_dict: Dict[DictKey,pyvips.Image]):
        array_dict = {k:np.array(v) for k,v in chunk_dict.items()}
        chunk_dict = {k:v for k,v in chunk_dict.items() if array_dict[v].std()>0}
        return chunk_dict
    
    def save_chunk(self, chunk:pyvips.Image, path:str):
        chunk.write_to_file(path)
        
    def run(self, path:str):
        img_path = Path(path)
        save_path = Path(self.save_folder) / img_path.stem
        save_path.mkdir(exist_ok=True, parents=True)
        img = self.load_image(path)
        img = self.process_image(img)
        chunks = self.split_image(img)
        chunks = self.filter_chunks(chunks)
        for (i,j), chunk in chunks.items():
            self.save_chunk(chunk, save_path / f'{i}_{j}.png')
        return chunks
            