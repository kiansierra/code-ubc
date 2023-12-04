# TODO List

## Future Work
- [ ] Mask Segmentation with custom backbone that can be reused for classification

## 2023-12-04
- [ ] Mixup
- [ ] CutMix



## 2023-12-01
- [ ] Validation label metrics
- [ ] Introduce other class
- [ ] Mask aware dataset

## 2023-29-11
- [x] Inference Script, takes in dataframe and checkpoint folder


## 2023-28-11
- [x] End to end preprocessing pipeline, using crop positions
- [ ] ~~ Compare component cropping time with and without given positions ~~



## 2023-25-11
- [x] Dataset configurations implement in validation
- [x] Config Loader
- [x] TrainingPipeline end to end (Dataset sequence and image size sequence)

## 2023-24-11
- [x] Dataset configurations (To train on different datasets, thumbnail, resized, cropped, etc)


## 2023-23-11
- [x] Resize steps also for masks
- [x] Crop steps with coordinates
- [x] Tile with masks


## 2023-21-11
- [x] Create Masks Datasets

## 2023-20-11
- [ ] Implemented Aggredated Tile Metric
- [x] Implementing Checkpoint Validation
- [ ] ~~Implementing Preprocessing Pipeline (Passing Images instead of load save in each phase)~~
- [x] Normalize Crop positions to 0-1

## 2023-19-11
- [x] Create preprocessing Tile
- [x] Extracting preprocessing logic to ubc/preprocess

## 2023-18-11
- [x] Streamline preprocesseing configs all using hydra for dynamic naming
- [x] Use WANDB API to download artifact from specific run for lineage and validation
- [x] Implement tma weighted loss and metrics
- [x] Add Dataset Lineage

## 2023-17-11

- [ ] ~~Implement tma classification and weighted loss~~
- [x] Image Scaling Train Pipeline and previous checkpoint loading
