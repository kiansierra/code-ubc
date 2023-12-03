import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image

from ..data.constants import label2idxmask


def get_crop_positions(img:Image.Image, image_id:int, output_folder:str, min_pct:float=0.02) -> Tuple[pd.DataFrame, List[Image.Image]]:
    mask = np.max(np.array(img) > 0, axis=-1).astype(np.uint8)
    retval, labels = cv2.connectedComponents(mask)
    postions = []
    crops = []
    component = 0
    for i in range(1, retval):
        valid_pixels = (labels == i) * mask ==1
        img_pct = valid_pixels.sum() / np.prod(mask.shape)
        if img_pct < min_pct:
            continue
        y, x = np.where(valid_pixels)
        crops.append(img.crop((x.min(), y.min(), x.max(), y.max())))
        sy, ey = y.min() / img.size[1], y.max() / img.size[1]
        sx, ex = x.min() / img.size[0], x.max() / img.size[0]
        path = f"{output_folder}/{image_id}_{component}.png"
        postions.append({"image_id":image_id, "path":path, "component": component,
                         "sx": sx, "ex": ex, "sy": sy, "ey": ey})
        component += 1
    logger.debug(f"Cropped {image_id=} with {img.size=} into {component=}")
    return pd.DataFrame(postions), crops


def get_cropped_images(
    image: Image.Image, image_id: str, output_folder: str, th_area: int = 1000
) -> Tuple[pd.DataFrame, List[Image.Image]]:
    # Aspect ratio
    as_ratio = image.size[0] / image.size[1]
    crop_id = 0
    outputs = []
    images = []
    if as_ratio >= 1.5:
        # Crop
        mask = np.max(np.array(image) > 0, axis=-1).astype(np.uint8)
        retval, labels = cv2.connectedComponents(mask)
        logger.debug(f"Cropping {image_id} with {as_ratio=:.2f} and size {image.size}")
        if retval >= as_ratio:
            x, y = np.meshgrid(np.arange(image.size[0]), np.arange(image.size[1]))
            for label in range(1, retval):
                area = np.sum(labels == label)
                if area < th_area:
                    continue
                xs = x[labels == label]
                sx, ex = np.min(xs), np.max(xs)
                cx = (sx + ex) // 2
                crop_size = image.size[1]
                sx = max(0, cx - crop_size // 2)
                ex = min(sx + crop_size - 1, image.size[0] - 1)
                sx = ex - crop_size + 1
                sy, ey = 0, image.size[1] - 1
                save_path = f"{output_folder}/{image_id}_{crop_id}.png"
                images.append(image.crop((sx, sy, ex, ey)))
                sx, sy = sx / image.size[0], sy / image.size[1]
                ex, ey = ex / image.size[0], ey / image.size[1]
                outputs.append(
                    {
                        "image_id": image_id,
                        "path": save_path,
                        "component": crop_id,
                        "sx": sx,
                        "ex": ex,
                        "sy": sy,
                        "ey": ey,
                    }
                )
                crop_id += 1
        else:
            crop_size = image.size[1]
            for i in range(int(as_ratio)):
                sx = i * crop_size
                ex = (i + 1) * crop_size - 1
                sx, ex = sx / image.size[0], ex / image.size[0]
                sy, ey = 0, 1
                save_path = f"{output_folder}/{image_id}_{crop_id}.png"
                images.append(image)
                outputs.append(
                    {
                        "image_id": image_id,
                        "path": save_path,
                        "component": crop_id,
                        "sx": sx,
                        "ex": ex,
                        "sy": sy,
                        "ey": ey,
                    }
                )
                crop_id += 1
    else:
        # Not Crop (entire image)
        sx, ex, sy, ey = 0, 1, 0, 1
        save_path = f"{output_folder}/{image_id}_{crop_id}.png"
        images.append(image)
        outputs.append(
            {"image_id": image_id, "path": save_path, "component": crop_id, "sx": sx, "ex": ex, "sy": sy, "ey": ey}
        )
    df_crop = pd.DataFrame(outputs)
    return df_crop, images


def get_crops_from_data(
    image: Image.Image, data: pd.DataFrame, output_folder: str
) -> Tuple[pd.DataFrame, List[Image.Image]]:
    # Aspect ratio
    records = data.to_dict("records")
    new_records, new_images = [], []
    for record in records:
        sx, sy = int(record["sx"] * image.size[0]), int(record["sy"] * image.size[1])
        ex, ey = int(record["ex"] * image.size[0]), int(record["ey"] * image.size[1])
        save_path = f"{output_folder}/{record['image_id']}_{record['component']}.png"
        crop = image.crop((sx, sy, ex, ey))
        new_records.append({"image_id": record["image_id"], "path": save_path, "component": record["component"]})
        new_images.append(crop)
    return pd.DataFrame(new_records), new_images


def resize(image: Image.Image, scale: float, imgsize: int = 2048):
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
    w2 = int(w * ratio)
    h2 = int(h * ratio)
    image = image.resize((w2, h2))
    logger.debug(f"debug: ({w}, {h}) --> ({w2}, {h2})")
    return image


def tiler(img: Image.Image, tile_size: int, empty_threshold: float) -> Dict[Tuple[int, int], np.ndarray]:
    """Split image into tiles of size tile_size x tile_size filters out tiles with less than empty_threshold
    Args:
        img (Image.Image): PIL image
        tile_size (int): tile size
        empty_threshold (float): threshold for empty tiles
    Returns:
        Dict[np.ndarray]]: Dict of tiles with coordinates as keys
    """
    img_array = np.array(img)
    x_splits = list(range(tile_size, img_array.shape[0], tile_size))
    y_splits = list(range(tile_size, img_array.shape[1], tile_size))
    all_splits = [np.split(elem, y_splits, 1) for elem in np.split(img_array, x_splits, 0)]
    output = {}
    for i, elem in enumerate(all_splits):
        for j, elem2 in enumerate(elem):
            if (elem2 > 0).mean() < empty_threshold:
                continue
            output[(i * tile_size, j * tile_size)] = elem2
    return output


def get_mask_weights(mask: np.ndarray) -> np.ndarray:
    """Get mask weights for weighted loss
    Args:
        mask (np.ndarray): mask
    Returns:
        np.ndarray: mask weights
    """
    mask_weights = (mask > 0).astype(np.float32).mean(axis=(0, 1))
    return {k: mask_weights[v] for k, v in label2idxmask.items()}


def tile_func(
    img: Image.Image,
    img_id: int,
    component: int,
    tile_size: int,
    empty_threshold: float,
    output_folder: str,
    get_weights: bool = False,
    center_crop:bool = False,
):
    logger.debug(f"tiling {img_id=} with {img.size=} and {tile_size=}")
    if center_crop:
        x_len = (img.size[0] // tile_size)*tile_size
        y_len = (img.size[1] // tile_size)*tile_size
        x_offset = (img.size[0] - x_len) // 2
        y_offset = (img.size[1] - y_len) // 2
        
        img = img.crop((x_offset, y_offset,
                        x_offset + x_len, y_offset + y_len))
    tiles = tiler(img, tile_size, empty_threshold)
    logger.debug(f"got {len(tiles)=} tiles for {img_id=}")
    data = []
    os.makedirs(f"{output_folder}/{img_id}", exist_ok=True)
    for (i, j), tile in tiles.items():
        if get_weights:
            mask_data = get_mask_weights(tile)
        weight = (tile.max(2) > 0).mean()
        tile = Image.fromarray(tile)
        path = f"{output_folder}/{img_id}/{component}_{i}_{j}.png"
        tile.save(path)
        tile_data = {"image_id": img_id, "path": path, "component": component, "i": i, "j": j, "weight": weight}
        if get_weights:
            tile_data = {**tile_data, **mask_data}
        data.append(tile_data)
    return pd.DataFrame(data)
