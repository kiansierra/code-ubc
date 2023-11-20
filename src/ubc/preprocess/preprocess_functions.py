
import os
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image


def get_cropped_images(image:Image.Image, image_id:str, output_folder:str, th_area:int = 1000):
    # Aspect ratio
    as_ratio = image.size[0] / image.size[1]
    crop_id = 0
    outputs = []
    if as_ratio >= 1.5:
        # Crop
        mask = np.max( np.array(image) > 0, axis=-1 ).astype(np.uint8)
        retval, labels = cv2.connectedComponents(mask)
        logger.debug(f"Cropping {image_id} with {as_ratio=:.2f} and size {image.size}")
        if retval >= as_ratio:
            x, y = np.meshgrid( np.arange(image.size[0]), np.arange(image.size[1]) )
            for label in range(1, retval):
                area = np.sum(labels == label)
                if area < th_area:
                    continue
                xs, ys= x[ labels == label ], y[ labels == label ]
                sx, ex = np.min(xs), np.max(xs)
                cx = (sx + ex) // 2
                crop_size = image.size[1]
                sx = max(0, cx-crop_size//2)
                ex = min(sx + crop_size - 1, image.size[0]-1)
                sx = ex - crop_size + 1
                sy, ey = 0, image.size[1]-1
                save_path = f"{output_folder}/{image_id}_{crop_id}.png"
                image.crop((sx,sy,ex,ey)).save(save_path)
                sx, sy = sx / image.size[0], sy / image.size[1]
                ex, ey = ex / image.size[0], ey / image.size[1]
                outputs.append({'image_id': image_id, 'path':save_path, 'component': crop_id, 'sx': sx, 'ex': ex, 'sy': sy, 'ey': ey})
                crop_id +=1
        else:
            crop_size = image.size[1]
            for i in range(int(as_ratio)):
                sx = i * crop_size
                ex = (i+1) * crop_size - 1
                sx, ex = sx / image.size[0], ex / image.size[0]
                sy, ey = 0, 1
                save_path = f"{output_folder}/{image_id}_{crop_id}.png"
                image.save(save_path)
                outputs.append({'image_id': image_id, 'path':save_path, 'component': crop_id, 'sx': sx, 'ex': ex, 'sy': sy, 'ey': ey})
                crop_id +=1
    else:
        # Not Crop (entire image)
        sx, ex, sy, ey = 0, 1, 0, 1
        save_path = f"{output_folder}/{image_id}_{crop_id}.png"
        image.save(save_path)
        outputs.append({'image_id': image_id, 'path':save_path, 'component': crop_id, 'sx': sx, 'ex': ex, 'sy': sy, 'ey': ey})
    df_crop = pd.DataFrame(outputs)
    return df_crop


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