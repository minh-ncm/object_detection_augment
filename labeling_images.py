import os
import glob
import cv2.cv2 as cv
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import convert_to_corners, unscale_boxes


def merge_labels():
    label_dir = '../all_ds/dataset/labels'
    labels = glob.glob('labelImg/*.txt')[:-1]

    for label in tqdm(labels,
                      desc='Merging'):
        new = open(label, 'r')
        name = label.split(os.sep)[-1]
        old = open(os.path.join(label_dir, name), 'a')
        old.writelines(new.readlines())

        old.close()
        new.close()


def delete_labelImg():
    files = glob.glob('labelImg/*')[:-1]
    for file in files:
        os.remove(file)


def delete_folder(path):
    if os.path.exists(path):
        files = glob.glob(path+'/*')
        for f in files:
            os.remove(f)


def draw_boxes_and_saves_images():
    images = glob.glob('dataset/images/*')
    labels = glob.glob('dataset/labels/*')
    images = glob.glob('processed/images/*')
    labels = glob.glob('processed/labels/*')

    label_names = ['no_mask', 'mask', 'incorrect_mask']
    max_sample = 10
    for i, (image, label) in enumerate(tqdm(zip(images, labels),
                                            total=len(images),
                                            desc='Labeling')):
        if i == max_sample:
            break
        try:
            img = cv.imread(image, cv.IMREAD_COLOR)
            h, w, c = img.shape
            name = os.path.split(image)[1]
            path = os.path.join('sample_original', name)
            path = os.path.join('sample_processed', name)

            label = pd.read_csv(label, sep=" ", header=None)
            label = label.to_numpy()
            for l in label:
                cls = np.array(l[0], np.int32)
                box = l[1:]
                box = convert_to_corners(box)
                box = unscale_boxes(box, (w, h))
                if cls == 0:
                    color = (0, 0, 255)
                elif cls == 1:
                    color = (0, 255, 0)
                else:
                    color = (0, 255, 255)
                img = cv.rectangle(img, pt1=box[:2], pt2=box[2:], color=color)
                img = cv.putText(img,
                                 text=label_names[cls],
                                 org=box[:2],
                                 fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                 fontScale=0.5,
                                 color=(255, 255, 255))

            cv.imwrite(path, img)
            # uncomment bellow lines to manually check each image
            # cv.imshow(f'{os.path.split(image)[1]}', img)
            # cv.waitKey(0)
            # cv.destroyWindow(f'{os.path.split(image)[1]}')
        except Exception as e:
            print('\n', e)
            print(os.path.split(image)[1])
        finally:
            cv.imwrite(path, img)


# merge_labels()
draw_boxes_and_saves_images()
# delete_labelImg()




