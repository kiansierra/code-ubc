import cv2
import numpy as np
from PIL import Image

__all__ = ["extract_components_positions"]


def extract_components_positions(path, area_threshold: int = 1_000, aspect_ratio_threshold: float = 1.5):
    img = Image.open(path)
    as_ratio = img.size[0] / img.size[1]
    if as_ratio <= aspect_ratio_threshold:
        return [(0, (img.size[0] - 1)/ img.size[0], 0, (img.size[1] - 1)/img.size[1])]
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
