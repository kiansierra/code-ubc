import albumentations as A
from albumentations.pytorch import ToTensorV2

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

__all__ = ["get_train_transforms", "get_valid_transforms"]


def get_train_transforms(config):
    return A.Compose(
        [
            A.Resize(config["img_size"], config["img_size"]),
            A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
    )


def get_valid_transforms(config):
    return A.Compose(
        [
            A.Resize(config["img_size"], config["img_size"]),
            A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
    )
