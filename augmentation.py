import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import skimage

from utils import *


class RandomFlip:
    def __init__(self, images=None, labels=None):
        """

        :param images: PIL Image
        :param labels: numpy array
        """
        self.__image = images
        self.__labels = labels

    def __flip_horizontal(self, image, labels):
        rand = np.random.randint(0, 2)
        if rand == 1:
            boxes = labels[..., 1:]
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            boxes[..., 0] = 1 - boxes[..., 0]
            labels = np.concatenate(
                [np.expand_dims(labels[..., 0], 1), boxes], axis=-1)
        return image, labels

    def __flip_vertical(self, image, labels):
        rand = np.random.randint(0, 2)
        if rand == 1:
            boxes = labels[..., 1:]
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            boxes[..., 1] = 1 - boxes[..., 1]
            labels = np.concatenate(
                [np.expand_dims(labels[..., 0], 1), boxes],
                axis=-1)
        return image, labels

    def augment(self, image, labels):
        self.__init__(image, labels)
        image, labels = self.__image, self.__labels

        image, labels = self.__flip_vertical(image, labels)
        image, labels = self.__flip_horizontal(image, labels)
        return image, labels


class RandomBlurBrightnessContrastNoise:
    def __init__(self, image=None, chance=1):
        """

        :param image: PIL Image
        :param chance:
        """
        self.__image = image
        self.__chance = chance
        self.__mode = [
            'gaussian',
            'localvar',
            'poisson',
            'salt',
            'pepper',
            's&p',
            'speckle'
        ]

    def __contrast(self, image):
        if np.random.uniform() <= self.__chance:
            image_enhance = ImageEnhance.Contrast(image)
            image = image_enhance.enhance(np.random.uniform(1, 2.5))
        return image

    def __brightness(self, image):
        if np.random.uniform() >= self.__chance:
            image_enhance = ImageEnhance.Brightness(image)
            image = image_enhance.enhance(np.random.uniform(0.2, 3))
        return image

    def __blur(self, image):
        if np.random.uniform() <= self.__chance:
            image = image.filter(ImageFilter.GaussianBlur(np.random.randint(1, 6)))
        return image

    def __noise(self, image):
        if np.random.uniform() >= self.__chance:
            image = np.array(image)
            mode = self.__mode[np.random.randint(len(self.__mode))]
            image = skimage.util.random_noise(image, mode) * 255
            image = Image.fromarray(np.array(image, dtype=np.uint8))
        return image

    def augment(self, image, chance=None):
        self.__image = image
        if chance:
            self.__chance = chance

        image = self.__brightness(image)
        image = self.__contrast(image)
        image = self.__blur(image)
        image = self.__noise(image)
        return image


class RandomCrop:
    def __init__(self, image=None, labels=None, require_class=None, low=0, high=1):
        self.__image = image
        self.__labels = labels
        self.__required_class = require_class
        self.__low = low
        self.__high = high

    def __relocate_box(self, label_box, crop_box, position):
        """

        :param label_box: label box (xmin, ymin, xmax, ymax)
        :param crop_box: crop window (xmin, ymin, xmax, ymax)
        :param position:
        :return:
        """
        new_box = swap_xy(label_box)
        new_box = np.concatenate([new_box[:2] - crop_box[0], new_box[2:] - crop_box[1]],
                                 axis=-1)

        size = crop_box[2:] - crop_box[:2]
        new_box = np.concatenate([new_box[:2] / size[0], new_box[2:] / size[1]])
        new_box = swap_xy(new_box)
        if position == 1:
            return new_box
        else:
            return np.clip(new_box, 0, 1)

    def __generate_window(self):
        rand_label_idx = np.random.choice(range(len(self.__labels)))
        if self.__required_class:
            while self.__labels[rand_label_idx, 0] != self.__required_class:
                rand_label_idx = np.random.choice(range(len(self.__labels)))

        start_time = end_time = time.time()
        while True:
            if end_time - start_time > 3:
                window_box = np.array([0, 0, 1, 1])
                break
            rand_size = np.random.uniform(self.__low, self.__high, (2,))
            rand_center = np.random.uniform(rand_size / 2, 1 - rand_size / 2,
                                            (2,))
            window_box = np.concatenate([rand_center, rand_size],
                                        axis=-1)
            window_box = convert_to_corners(window_box)
            position = check_position(
                convert_to_corners(self.__labels[rand_label_idx, 1:]), window_box)
            if position == 1:
                break
            else:
                end_time = time.time()
        return window_box

    def __crop(self):
        window_box = self.__generate_window()
        new_label = []
        for label in self.__labels:
            label_box = convert_to_corners(label[1:])
            position = check_position(label_box, window_box)
            if position != -1:
                box = self.__relocate_box(label_box, window_box, position)
                if position == 0:
                    old_area = calc_area(box)
                    box = np.clip(box, 0, 1)
                    new_area = calc_area(box)
                    if old_area == 0 or new_area / old_area <= 0.5:
                        break
                box = convert_to_xywh(box)
                new_label.append(
                    np.concatenate([label[np.newaxis, 0], box], axis=-1))

        window_box = unscale_boxes(window_box, self.__image.size)
        window = self.__image.crop(window_box)
        window = window.resize(self.__image.size, resample=Image.LANCZOS)
        return window, np.array(new_label)

    def augment(self, image, labels, require_class=None):
        self.__image, self.__labels, self.__required_class = image, labels, require_class

        if np.random.uniform() >= 0.8:
            return self.__crop()
        else:
            return self.__image, self.__labels


class RandomRotate:
    def __init__(self, image=None, labels=None, require_class=None):
        self.__image = image
        self.__labels = labels
        self.__require_class = require_class

    def __rotate_label(self, alpha, image_size):
        """
        Counter-clockwise rotation
        :param alpha:
        :param image_size:
        :return:
        """
        center = image_size / 2
        alpha *= (np.pi / 180)
        rotation_matrix = np.array([
            [np.cos(alpha), np.sin(alpha)],
            [-np.sin(alpha), np.cos(alpha)]
        ])
        new_labels = []
        boxes = convert_to_corners(unscale_boxes(self.__labels[..., 1:], image_size))
        classes = np.expand_dims(self.__labels[..., 0], -1)

        for box, cls in zip(boxes, classes):
            new_box = np.empty((0, 2), dtype=np.float32)
            box = get_all_corners(box)
            for point in box:
                point = point - center
                point = np.matmul(rotation_matrix, point)
                point = np.expand_dims(point + center, 0)
                new_box = np.append(new_box, point, axis=0)
            new_box = np.concatenate([np.min(new_box, axis=0), np.max(new_box, axis=0)])
            new_box.resize((4,))
            new_box = rescale_boxes(new_box, image_size)
            position = check_position(new_box, np.array([0, 0, 1, 1]))
            if position != -1:
                if position == 0:
                    old_area = calc_area(new_box)
                    new_box = np.clip(new_box, 0, 1)
                    new_area = calc_area(new_box)
                    if old_area == 0 or new_area / old_area <= 0.5:
                        break
            else:
                break
            new_box = convert_to_xywh(new_box)
            new_labels.append(np.concatenate([cls, new_box], axis=-1))
        return np.array(new_labels)

    def augment(self, image, labels, require_class=None):
        self.__image, self.__labels, self.__required_class = image, labels, require_class

        rotated_labels = []
        start_time = end_time = time.time()
        while len(rotated_labels) == 0:
            alpha = np.arange(-45, 46)
            alpha = int(np.random.normal(alpha.mean(), alpha.std()) * 45)
            if end_time - start_time > 3:
                alpha = 0
            rotated_image = self.__image.rotate(alpha, resample=Image.BICUBIC)
            rotated_labels = self.__rotate_label(alpha, np.array(self.__image.size))
            end_time = time.time()

        return rotated_image, rotated_labels


class RandomCutmix:
    def __init__(self, image_pool, image=None, labels=None):
        self.__image = image
        self.__labels = labels
        self.__image_pool = image_pool
        self.__nw_pool, self.__w_pool, self.__iw_pool = get_path_for_each_class(image_pool[..., 0], image_pool[..., 1], verbose=False)
        self.__random_crop = RandomCrop(low=0.3, high=0.6)
        self.__random_rotate = RandomRotate()

    def __remove_overlap_bbox(self, labels, window_box):
        new_labels = []
        for label in labels:
            bbox = label[..., 1:]
            cls = np.expand_dims(label[..., 0], axis=-1)
            bbox_corners = convert_to_corners(bbox)
            window_box_corners = convert_to_corners(window_box)
            position = check_position(bbox_corners, window_box_corners)
            if position != 1:
                if position == 0:
                    bbox_area = bbox[2] * bbox[3]
                    tl = np.maximum(bbox_corners[:2], window_box_corners[:2])
                    br = np.minimum(bbox_corners[2:], window_box_corners[2:])
                    intersection = np.maximum(0, br - tl)
                    intersect_area = intersection[0] * intersection[1]
                    if bbox_area != 0 and intersect_area / bbox_area < 0.5:
                        new_labels.append(np.concatenate([cls, bbox]))
                else:
                    new_labels.append(label)
        new_labels = np.array(new_labels)
        return new_labels

    def __relocate_labels(self, cutout_labels, cutout_window, image_size):
        """

        :param cutout_labels:
        :param cutout_window: (x, y, w, h)
        :param image_size:
        :return:
        """
        unscale_cutout = unscale_boxes(cutout_window, image_size)
        cutout_bbox = convert_to_corners(cutout_labels[..., 1:])
        cutout_bbox = unscale_boxes(cutout_bbox, unscale_cutout[2:])
        unscale_cutout = convert_to_corners(unscale_cutout)
        relocated_labels = swap_xy(cutout_bbox)
        relocated_labels = np.concatenate([
            relocated_labels[..., :2] + unscale_cutout[0],
            relocated_labels[..., 2:] + unscale_cutout[1],
        ], axis=-1)
        relocated_labels = swap_xy(relocated_labels)
        relocated_labels = convert_to_xywh(relocated_labels)
        relocated_labels = np.concatenate([
            np.expand_dims(cutout_labels[..., 0], axis=-1),
            rescale_boxes(relocated_labels, image_size)
        ], axis=-1)
        return relocated_labels

    def __cutmix_new_targets(self):
        cutout_num = np.random.randint(1, 4)
        cutout_windows = np.concatenate([
            np.random.uniform(0, 1, size=(cutout_num, 2)),
            np.random.uniform(0.2, 0.4, size=(cutout_num, 2))
        ], axis=-1)
        cutout_windows = np.clip(convert_to_corners(cutout_windows), 0, 1)

        image = self.__image
        new_labels = self.__labels
        for cutout_window in convert_to_xywh(cutout_windows):
            new_labels = self.__remove_overlap_bbox(new_labels,
                                                    cutout_window)
            tmp = np.random.uniform()
            if tmp < 0.1:
                cutout_class = 1
                image_pool = self.__w_pool
            elif tmp < 0.4:
                cutout_class = 0
                image_pool = self.__nw_pool
            else:
                cutout_class = 2
                image_pool = self.__iw_pool
            cutout_idx = np.random.choice(range(len(image_pool)))
            cutout_image = Image.open(image_pool[cutout_idx][0])
            cutout_label = pd.read_csv(image_pool[cutout_idx][1],
                                       header=None,
                                       sep=' ').to_numpy()

            cutout_image, cutout_label = self.__random_crop.augment(
                cutout_image,
                cutout_label,
                cutout_class)
            cutout_image, cutout_label = self.__random_rotate.augment(
                cutout_image, cutout_label)
            cutout_label = self.__relocate_labels(cutout_label, cutout_window, self.__image.size)
            if new_labels.size != 0:
                new_labels = np.concatenate([new_labels, cutout_label])
            else:
                new_labels = cutout_label
            cutout_window = np.array(transpose_xy(
                convert_to_corners(unscale_boxes(cutout_window, self.__image.size))),
                                  dtype=np.uint16)
            image_matrix = np.array(image)
            cutout_image = cutout_image.resize([cutout_window[3] - cutout_window[1], cutout_window[2] - cutout_window[0]])
            image_matrix[cutout_window[0]: cutout_window[2], cutout_window[1]: cutout_window[3]] = np.array(cutout_image)
            image = Image.fromarray(image_matrix)
        return image, new_labels

    def augment(self, image, labels):
        self.__image, self.__labels = image, labels

        image, labels = self.__cutmix_new_targets()
        return image, labels