import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

__all__ = ["get_train_transforms", "get_valid_transforms"]


def get_blurs():
    blurs = [A.Blur(), A.GaussianBlur(), A.MotionBlur(), A.MedianBlur()]
    return A.OneOf(blurs, p=0.5)


def get_dropouts():
    dropouts = [A.Cutout(num_holes=10, max_h_size=16, max_w_size=16), A.PixelDropout(dropout_prob=0.05)]
    return A.OneOf(dropouts, p=0.5)


def get_train_transforms(config: DictConfig) -> A.Compose:
    transforms = [
        A.Resize(config.img_size, config.img_size),
        A.HorizontalFlip(p=0.5) if config.get("flip", False) else None,
        A.VerticalFlip(p=0.5) if config.get("flip", False) else None,
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        get_blurs() if config.get("blur", False) else None,
        get_dropouts() if config.get("dropout", False) else None,
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD, max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
    ]
    transforms = [transform for transform in transforms if transform is not None]
    return A.Compose(transforms, p=1.0)


def get_valid_transforms(config: DictConfig) -> A.Compose:
    return A.Compose(
        [
            A.Resize(config.img_size, config.img_size),
            A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ],
        p=1.0,
    )
