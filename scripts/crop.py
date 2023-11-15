import argparse
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image


def get_cropped_images(file_path, image_id, output_folder, th_area = 1000):
    image = Image.open(file_path)
    # Aspect ratio
    as_ratio = image.size[0] / image.size[1]
    crop_id = 0
    sxs, exs, sys, eys, file_paths = [],[],[],[], []
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
                sxs.append(sx)
                exs.append(ex)
                sys.append(sy)
                eys.append(ey)
                save_path = f"{output_folder}/{image_id}_{crop_id}.png"
                image.crop((sx,sy,ex,ey)).save(save_path)
                file_paths.append(save_path)
                crop_id +=1
        else:
            crop_size = image.size[1]
            for i in range(int(as_ratio)):
                sxs.append( i * crop_size )
                exs.append( (i+1) * crop_size - 1 )
                sys.append( 0 )
                eys.append( crop_size - 1 )
                save_path = f"{output_folder}/{image_id}_{crop_id}.png"
                image.save(save_path)
                file_paths.append(save_path)
                crop_id +=1
    else:
        # Not Crop (entire image)
        sxs, exs, sys, eys = [0,],[image.size[0]-1],[0,],[image.size[1]-1]
        save_path = f"{output_folder}/{image_id}_{crop_id}.png"
        image.save(save_path)
        file_paths.append(save_path)

    df_crop = pd.DataFrame()
    df_crop["image_id"] = [image_id] * len(sxs)
    df_crop["image_size"] = [image.size] * len(sxs)
    df_crop["file_path"] = file_paths
    df_crop["sx"] = sxs
    df_crop["ex"] = exs
    df_crop["sy"] = sys
    df_crop["ey"] = eys
    return df_crop


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-folder',   type=str,   default="output file path")
    parser.add_argument('--dataframe-path',   type=str,   default="Dataframe path")
    parser.add_argument('--num_processes',   type=int,   default=4)
    args = parser.parse_args()
    return args

def crop(row, output_folder):
    return get_cropped_images(row['path'], row['image_id'], output_folder)


def main():
    args = args_parser()
    df = pd.read_parquet(args.dataframe_path)
    records = df.to_dict('records')
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    crop_save = partial(crop, output_folder=output_folder)
    if args.num_processes > 1:
        with mp.Pool(args.num_processes) as pool:
            outputs = pool.map(crop_save, records)
    else:
        outputs = list(map(crop_save, records))
    output_df = pd.concat(outputs)
    output_df['component'] = output_df.index
    output_df.reset_index(drop=True, inplace=True)
    output_df = output_df.merge(df, on='image_id', how='left')
    output_df.to_parquet(f"{args.output_folder}/train.parquet")
    
if __name__ == '__main__':
    main()