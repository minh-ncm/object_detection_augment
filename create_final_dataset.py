import glob
import os
import shutil
import sys
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import utils
from augmentation import *


src_image_path = np.array(glob.glob('dataset/images/*'))
src_label_path = np.array(glob.glob('dataset/labels/*'))

processed_image_dir = os.path.join('processed', 'images')
processed_label_dir = os.path.join('processed', 'labels')
Path(processed_image_dir).mkdir(parents=True, exist_ok=True)
Path(processed_label_dir).mkdir(parents=True, exist_ok=True)

data_pool = [[image_path, label_path] for image_path, label_path in zip(
    src_image_path,
    src_label_path
)]
total = len(data_pool)

with tqdm(total=total, desc='Creating dataset') as pbar:
    for i, data in enumerate(data_pool):
        image_path = data[0]
        label_path = data[1]

        # cutmix = RandomCutmix(image_pool)
        crop = RandomCrop(0.5, 0.9)
        rotate = RandomRotate()
        flip = RandomFlip()
        blur_brightness_contrast_noise = RandomBlurBrightnessContrastNoise(chance=0.7)

        image = Image.open(image_path)
        label = pd.read_csv(label_path, sep=' ', header=None).to_numpy()
        image, label = crop.augment(image, label)
        image, label = rotate.augment(image, label)
        # image, label = cutmix.augment(image, label)
        image, label = flip.augment(image, label)
        image = blur_brightness_contrast_noise.augment(image)

        name = os.path.split(image_path)[-1]
        name = name.split('/')[0]
        image.save(os.path.join(processed_image_dir, name+'.jpg'))
        np.savetxt(os.path.join(processed_label_dir, name+'.txt'), label, fmt='%.6f')
        pbar.update(1)


# print('Compressing...')
# zip_dir = os.path.join(final_dir, 'zip')
# id = len(os.listdir(zip_dir))
# shutil.make_archive(zip_dir+f'/dataset{id:02d}',
#                     format='zip',
#                     root_dir=final_dir,
#                     base_dir='dataset')
# print('Done!')