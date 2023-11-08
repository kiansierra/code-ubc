from typing import Dict, Protocol

import numpy as np
import pyvips


class PreProcessor(Protocol):
    
    def load_image(self, path:str):
        ...
        
    def process_image(self, img:np.ndarray | pyvips.Image):
        ...
        
    def split_image(self, img:np.ndarray | pyvips.Image):
        ...
        
    def filter_chunks(self, img:Dict[np.ndarray] | Dict[pyvips.Image]):
        ...
            
    def run(self, path:str):
        img = self.load_image(path)
        img = self.process_image(img)
        chunks = self.split_image(img)
        chunks = self.filter_chunks(chunks)
        return chunks
        
    