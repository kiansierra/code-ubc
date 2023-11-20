
import os
from loguru import logger
from PIL import Image
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def resize(image: Image.Image, scale:float, imgsize:int=2048):
    """
    Resize image to imgsize with scale factor and keep aspect ratio
    Args:
        image (Image.Image): PIL image
        scale (float): scale factor
        imgsize (int, optional): minimum image size. Defaults to 2048.
    Returns:
        Image.Image: resized image
    """
    w, h = image.size
    ratio = imgsize / min([w, h])
    ratio = max(ratio, scale)
    w2 = int(w*ratio)
    h2 = int(h*ratio)
    image = image.resize( (w2, h2) )
    logger.debug(f"debug: ({w}, {h}) --> ({w2}, {h2})")
    return image

def tiler(img: Image.Image, tile_size: int, empty_threshold:float) -> Dict[Tuple[int,int],np.ndarray]:
    """Split image into tiles of size tile_size x tile_size filters out tiles with less than empty_threshold
    Args:
        img (Image.Image): PIL image
        tile_size (int): tile size
        empty_threshold (float): threshold for empty tiles
    Returns:
        Dict[np.ndarray]]: Dict of tiles with coordinates as keys
    """
    img_array = np.array(img)
    x_splits = list(range(tile_size,img_array.shape[0],tile_size))
    y_splits = list(range(tile_size,img_array.shape[1],tile_size))
    all_splits = [np.split(elem, y_splits, 1) for elem in np.split(img_array, x_splits, 0)]
    output = {}
    for i, elem in enumerate(all_splits):
        for j, elem2 in enumerate(elem):
            if (elem2>0).mean()<empty_threshold:
                continue
            output[(i*tile_size,j*tile_size)] = elem2
    return output

def tile_func(img:Image.Image, img_id:int, component:int, tile_size:int, empty_threshold:float, output_folder:str):
    logger.debug(f"tiling {img_id} with {img.size} and {tile_size=}")
    tiles = tiler(img, tile_size, empty_threshold)
    logger.debug(f"got {len(tiles)} tiles for {img_id=}")
    data = []
    os.makedirs(f"{output_folder}/{img_id}", exist_ok=True)
    for (i,j), tile in tiles.items():
        tile = Image.fromarray(tile)
        path = f"{output_folder}/{img_id}/{component}_{i}_{j}.png"
        tile.save(path)
        data.append({'image_id': img_id, 'path':path, 'component': component, 'i': i, 'j': j})
    return pd.DataFrame(data)