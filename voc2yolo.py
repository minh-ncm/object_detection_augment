from lxml import etree
import glob
import os
from tqdm import tqdm
import numpy as np

import utils


voc_anno_paths = glob.glob('../all_ds/additions/annotations/*')
yolo_anno_dir = '../all_ds/additions/add01/labels'

for path in tqdm(voc_anno_paths,
                 desc='cConverting VOC to YOLO format'):
    tree = etree.parse(path)
    root = tree.getroot()
    objects = root.xpath('./object')
    image_name = root.xpath('./filename[1]')[0].text
    image_size = np.array([
        int(root.xpath('./size/width')[0].text),
        int(root.xpath('./size/height')[0].text)
    ])

    classes = []
    boxes = []
    for obj in objects:
        cls = obj.xpath('./name')[0].text
        if cls == 'without_mask':
            cls = 0
        elif cls == 'with_mask':
            cls = 1
        else:
            cls = 2
        box = [
            int(obj.xpath('./bndbox/xmin')[0].text),
            int(obj.xpath('./bndbox/ymin')[0].text),
            int(obj.xpath('./bndbox/xmax')[0].text),
            int(obj.xpath('./bndbox/ymax')[0].text)
        ]
        classes.append(cls)
        boxes.append(box)

    boxes = np.array(boxes)
    boxes = utils.rescale_boxes(boxes, image_size)
    boxes = utils.convert_to_xywh(boxes)
    classes = np.array(classes)[..., np.newaxis]
    labels = np.concatenate([classes, boxes], axis=-1)
    np.savetxt(os.path.join(yolo_anno_dir, image_name.split('.')[0]+'.txt'),
               labels,
               fmt='%.6f')