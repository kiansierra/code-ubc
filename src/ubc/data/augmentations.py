from typing import Callable

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig

from ..utils import Registry
from .custom_augmentations import CustomRandomSizedCropNoResize

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

__all__ = [
    "get_train_transforms",
    "get_valid_transforms",
    "get_train_basic_transform",
    "AUGMENTATIONS_REGISTRY",
    "build_augmentations",
]

AugmentationBuilderType = Callable[[DictConfig], A.Compose]
AUGMENTATIONS_REGISTRY = Registry("AUGMENTATIONS", AugmentationBuilderType)


def build_augmentations(config: DictConfig) -> A.Compose:
    return AUGMENTATIONS_REGISTRY.get(config.get("name", "get_valid_transforms"))(config)


def get_blurs():
    blurs = [A.Blur(), A.GaussianBlur(), A.MotionBlur(), A.MedianBlur()]
    return A.OneOf(blurs, p=0.5)


def get_dropouts():
    dropouts = [A.Cutout(num_holes=10, max_h_size=16, max_w_size=16), A.PixelDropout(dropout_prob=0.05)]
    return A.OneOf(dropouts, p=0.5)

def get_resize(img_size:int, method:str):
    if method == 'resize':
        return A.Resize(img_size, img_size)
    elif method == 'padded':
        return A.Compose([A.LongestMaxSize(img_size), A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT)])
    elif method == 'crop':
        return A.RandomCrop(img_size, img_size)
    elif method == 'mixed':
        return A.OneOf([
            A.Resize(img_size, img_size),
            A.Compose([A.LongestMaxSize(img_size), A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT)]),
            A.RandomCrop(img_size, img_size),
        ])


@AUGMENTATIONS_REGISTRY.register()
def get_train_transforms(config: DictConfig) -> A.Compose:
    transforms = [
        get_resize(config.img_size, config.get("resize_method", "resize")),
        A.HorizontalFlip(p=0.5) if config.get("flip", False) else None,
        A.VerticalFlip(p=0.5) if config.get("flip", False) else None,
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        get_blurs() if config.get("blur", False) else None,
        get_dropouts() if config.get("dropout", False) else None,
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD, max_pixel_value=255.0, p=1.0),
        ToTensorV2(transpose_mask=True),
    ]
    transforms = [transform for transform in transforms if transform is not None]
    return A.Compose(transforms, p=1.0)


@AUGMENTATIONS_REGISTRY.register()
def get_train_basic_transform(config: DictConfig) -> A.Compose:
    return A.Compose(
        [
            get_resize(config.img_size, config.get("resize_method", "resize")),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD, max_pixel_value=255.0, p=1.0),
            ToTensorV2(transpose_mask=True),
        ],
        p=1.0,
    )


@AUGMENTATIONS_REGISTRY.register()
def get_train_rsna_transform(config: DictConfig) -> A.Compose:
    return A.Compose(
        [
            # crop, tweak from A.RandomSizedCrop()
            CustomRandomSizedCropNoResize(scale=(0.5, 1.0), ratio=(0.5, 0.8), p=0.4),
            # flip
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # downscale
            A.OneOf(
                [
                    A.Downscale(
                        scale_min=0.75,
                        scale_max=0.95,
                        interpolation=dict(upscale=cv2.INTER_LINEAR, downscale=cv2.INTER_AREA),
                        p=0.1,
                    ),
                    A.Downscale(
                        scale_min=0.75,
                        scale_max=0.95,
                        interpolation=dict(upscale=cv2.INTER_LANCZOS4, downscale=cv2.INTER_AREA),
                        p=0.1,
                    ),
                    A.Downscale(
                        scale_min=0.75,
                        scale_max=0.95,
                        interpolation=dict(upscale=cv2.INTER_LINEAR, downscale=cv2.INTER_LINEAR),
                        p=0.8,
                    ),
                ],
                p=0.125,
            ),
            # contrast
            A.OneOf(
                [
                    A.RandomToneCurve(scale=0.3, p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.1, 0.2),
                        contrast_limit=(-0.4, 0.5),
                        brightness_by_max=True,
                        always_apply=False,
                        p=0.5,
                    ),
                ],
                p=0.5,
            ),
            # geometric
            A.OneOf(
                [
                    A.ShiftScaleRotate(
                        shift_limit=None,
                        scale_limit=[-0.15, 0.15],
                        rotate_limit=[-30, 30],
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=None,
                        shift_limit_x=[-0.1, 0.1],
                        shift_limit_y=[-0.2, 0.2],
                        rotate_method="largest_box",
                        p=0.6,
                    ),
                    A.ElasticTransform(
                        alpha=1,
                        sigma=20,
                        alpha_affine=10,
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=None,
                        approximate=False,
                        same_dxdy=False,
                        p=0.2,
                    ),
                    A.GridDistortion(
                        num_steps=5,
                        distort_limit=0.3,
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=None,
                        normalized=True,
                        p=0.2,
                    ),
                ],
                p=0.5,
            ),
            # Random Crop
            A.RandomCrop(config["img_size"], config["img_size"]),
            # random erase
            A.CoarseDropout(
                max_holes=6,
                max_height=0.15,
                max_width=0.25,
                min_holes=1,
                min_height=0.05,
                min_width=0.1,
                fill_value=0,
                mask_fill_value=None,
                p=0.25,
            ),
        ],
        p=0.9,
    )


@AUGMENTATIONS_REGISTRY.register()
def get_valid_transforms(config: DictConfig) -> A.Compose:
    return A.Compose(
        [
            get_resize(config.img_size, config.get("resize_method", "resize")),
            A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD, max_pixel_value=255.0, p=1.0),
            ToTensorV2(transpose_mask=True),
        ],
        p=1.0,
    )
