import os, glob, time
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def display_image_with_labels(image, labels):
    """

    :param image: PIl Image
    :param labels: numpy arrays, (0, 1) range
    :return:
    """
    classes = labels[..., 0]
    boxes = convert_to_corners(labels[..., 1:])
    boxes = unscale_boxes(boxes, image.size)
    draw = ImageDraw.Draw(image)
    for cls, box in zip(classes, boxes):
        if cls == 0:
            color = (255, 0, 0)
        elif cls == 1:
            color = (0, 255, 0)
        else:
            color = (255, 255, 0)
        draw.rectangle([tuple(box[:2]), tuple(box[2:])], outline=color)
        draw.line([tuple(box[:2]), tuple(box[2:])], fill=color)
    image.show()


def convert_to_corners(boxes):
    """

    :param boxes: (x, y, width, height)
    :return:
    box: (xmin, ymin, xmax, ymax)
    """
    return np.concatenate([boxes[..., :2] - boxes[..., 2:] / 2,
                           boxes[..., :2] + boxes[..., 2:] / 2],
                          axis=-1)


def convert_to_xywh(boxes):
    """

    :param boxes: (xmin, ymin, xmax, ymax)
    :return: (x, y, width, height)
    """
    return np.concatenate([(boxes[..., :2] + boxes[..., 2:]) / 2,
                           (boxes[..., 2:] - boxes[..., :2])],
                          axis=-1)


def swap_xy(boxes):
    """
    swap ymin, xmax
    :param boxes:
    :return:
    """
    return np.stack(
        [boxes[..., 0], boxes[..., 2], boxes[..., 1], boxes[..., 3]], axis=-1)


def transpose_xy(boxes):
    return np.stack(
        [boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1)


def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
    boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
    boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = np.maximum(boxes1_corners[:2], boxes2_corners[:2])
    rd = np.minimum(boxes1_corners[2:], boxes2_corners[2:])
    intersection = np.maximum(0.0, rd - lu)
    intersection_area = intersection[0] * intersection[1]
    boxes1_area = boxes1[2] * boxes1[3]
    boxes2_area = boxes2[2] * boxes2[3]
    union_area = np.maximum(
        boxes1_area + boxes2_area - intersection_area, 1e-8
    )
    return np.clip(intersection_area / union_area, 0.0, 1.0)


def get_all_corners(box):
    """

    :param box: (xmin, ymin, xmax, ymax)
    :return:
    """
    return np.array([
        box[:2],
        [box[2], box[1]],
        box[2:],
        [box[0], box[3]]
    ])


def unscale_boxes(boxes, image_size):
    """

    :param boxes: (xmin, ymin, xmax, ymax), [0-1]
    :param image_size: (width, height)
    :return:
    """
    boxes = swap_xy(boxes)
    boxes = np.concatenate(
        [boxes[..., :2] * image_size[0], boxes[..., 2:] * image_size[1]],
        axis=-1)
    return swap_xy(boxes.astype(np.int16))


def rescale_boxes(boxes, image_size):
    """

    :param boxes: (xmin, ymin, xmax, ymax)
    :param image_size:(width, height)
    :return:
    """
    boxes = swap_xy(boxes)
    boxes = np.concatenate(
        [boxes[..., :2] / image_size[0], boxes[..., 2:] / image_size[1]],
        axis=-1)
    return swap_xy(boxes)


def calc_area(boxes):
    """

    :param boxes: (xmin, ymin, xmax, ymax)
    :return:
    """
    e = boxes[..., 2:] - boxes[..., :2]
    return e[..., 0] * e[..., 1]


def check_position(box1, box2):
    """
    Return where box1 is located, relative to box2.
    :param box1: (xmin, ymin, xmax, ymax)
    :param box2: (xmin, ymin, xmax, ymax)
    :return: 1 if box1 inside box2, 0 if box1 overlap with box2's edges, -1 if box1 outside box2.
    """
    box2_max = box2[2:]
    box2_min = box2[:2]
    vertices = get_all_corners(box1)
    if (vertices[0] >= box2_min).all() and (vertices[2] <= box2_max).all():
        return 1
    for vertex in vertices:
        if (vertex <= box2_max).all() and (vertex >= box2_min).all():
            return 0
    return -1


def random_crop(image, labels, required_class=None, low=0, high=1):
    """

    :param image: PIL Image
    :param labels: numpy arrays
    :param required_class: must-have class after cropping.
    :param low: minimum crop size
    :param high: maximum crop size
    :return:
    """

    def _relocate_box(box1, box2, position):
        """

        :param box1: label box (xmin, ymin, xmax, ymax)
        :param box2: crop window (xmin, ymin, xmax, ymax)
        :param position:
        :return:
        """
        new_box = swap_xy(box1)
        new_box = np.concatenate([new_box[:2] - box2[0], new_box[2:] - box2[1]],
                                 axis=-1)

        size = box2[2:] - box2[:2]
        new_box = np.concatenate([new_box[:2] / size[0], new_box[2:] / size[1]])
        new_box = swap_xy(new_box)
        if position == 1:
            return new_box
        else:
            return np.clip(new_box, 0, 1)

    # generate window
    rand_label_idx = np.random.choice(range(len(labels)))
    if required_class:
        while labels[rand_label_idx, 0] != required_class:
            rand_label_idx = np.random.choice(range(len(labels)))

    start_time = end_time = time.time()
    while True:
        if end_time - start_time > 5:
            window_box = np.array([0, 0, 1, 1])
            break
        rand_size = np.random.uniform(low, high, (2,))
        rand_center = np.random.uniform(rand_size / 2, 1 - rand_size / 2, (2,))
        window_box = np.concatenate([rand_center, rand_size],
                                    axis=-1)
        window_box = convert_to_corners(window_box)
        position = check_position(convert_to_corners(labels[rand_label_idx, 1:]), window_box)
        if position == 1:
            break
        else:
            end_time = time.time()

    # relocate labels
    new_label = []
    for label in labels:
        label_box = convert_to_corners(label[1:])
        position = check_position(label_box, window_box)
        if position != -1:
            box = _relocate_box(label_box, window_box, position)
            if position == 0:
                old_area = calc_area(box)
                box = np.clip(box, 0, 1)
                new_area = calc_area(box)
                if new_area / old_area <= 0.4:
                    break
            box = convert_to_xywh(box)
            new_label.append(np.concatenate([label[np.newaxis, 0], box], axis=-1))

    window_box = unscale_boxes(window_box, image.size)
    window = image.crop(window_box)
    window = window.resize(image.size, resample=Image.LANCZOS)
    return window, np.array(new_label)


def random_rotate(image, labels, required_class=None):
    """

    :param image: PIL Image
    :param labels: numpy arrays
    :param required_class: must-have class after rotate
    :return:
    """
    def _rotate_label(labels, alpha, image_size):
        """
        Counter-clockwise rotation
        :param labels:
        :param alpha:
        :param image_size:
        :return:
        """
        labels_corners = labels.copy()
        labels_corners[..., 1:] = convert_to_corners(unscale_boxes(labels_corners[..., 1:], image.size))
        center = image_size / 2
        alpha *= (np.pi / 180)
        rotation_matrix = np.array([
            [np.cos(alpha), np.sin(alpha)],
            [-np.sin(alpha), np.cos(alpha)]
        ])
        new_labels = []
        boxes = labels_corners[..., 1:]
        classes = np.expand_dims(labels_corners[..., 0], -1)

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
            new_box = rescale_boxes(new_box, image.size)
            position = check_position(new_box, np.array([0, 0, 1, 1]))
            if position != -1:
                if position == 0:
                    old_area = calc_area(new_box)
                    new_box = np.clip(new_box, 0, 1)
                    new_area = calc_area(new_box)
                    if new_area / old_area <= 0.4:
                        break
            else:
                break
            new_box = convert_to_xywh(new_box)
            new_labels.append(np.concatenate([cls, new_box], axis=-1))
        return np.array(new_labels)

    rotated_labels = []
    start_time = end_time = time.time()
    rotated_image = image
    while len(rotated_labels) == 0:
        alpha = np.arange(-45, 46)
        alpha = int(np.random.normal(alpha.mean(), alpha.std()) * 45)
        if end_time - start_time > 5:
            alpha = 0
        rotated_image = image.rotate(alpha, resample=Image.BICUBIC)
        rotated_labels = _rotate_label(labels, alpha, np.array(image.size))
        end_time = time.time()

    return rotated_image, rotated_labels


def random_cutmix(image, labels, image_pool):
    def _cutmix_new_targets(image, labels, image_pool):
        def _relocate_labels(cutout_labels, cutout_box, image_size):
            """

            :param cutout_labels:
            :param cutout_box: (x, y, w, h)
            :param image_size:
            :return:
            """
            unscale_cutout = unscale_boxes(cutout_box, image_size)
            cutout_bbox = cutout_labels[..., 1:]
            cutout_bbox = convert_to_corners(cutout_bbox)
            cutout_bbox = unscale_boxes(cutout_bbox, unscale_cutout[2:])
            unscale_cutout = convert_to_corners(unscale_cutout)
            relocated_labels = swap_xy(cutout_bbox)
            relocated_labels = np.concatenate([
                relocated_labels[..., :2] + unscale_cutout[0],
                relocated_labels[..., 2:] + unscale_cutout[1],
            ], axis=-1)
            relocated_labels = swap_xy(relocated_labels)
            relocated_labels = convert_to_xywh(relocated_labels)
            relocated_labels = rescale_boxes(relocated_labels, image_size)
            relocated_labels = np.concatenate([
                np.expand_dims(cutout_labels[..., 0], axis=-1),
                relocated_labels
            ], axis=-1)
            return relocated_labels

        nw_glob, w_glob, iw_glob = get_path_for_each_class(image_pool[:, 0],
                                                           image_pool[:, 1])
        cutout_num = np.random.randint(3, 10)
        cutout_boxes = np.concatenate([
            np.random.uniform(0, 1, size=(cutout_num, 2)),
            np.random.uniform(0.05, 0.3, size=(cutout_num, 2))
        ], axis=-1)
        cutout_boxes = np.clip(convert_to_corners(cutout_boxes), 0, 1)

        cutout_labels = []
        for cutout_box in convert_to_xywh(cutout_boxes):
            if np.random.uniform() < 0.3:
                cutout_class = 0
                image_pool = nw_glob
            else:
                cutout_class = 2
                image_pool = iw_glob
            cutout_idx = np.random.choice(range(len(image_pool)))
            cutout_image = Image.open(image_pool[cutout_idx][0])
            cutout_label = pd.read_csv(image_pool[cutout_idx][1],
                                       header=None,
                                       sep=' ').to_numpy()
            cutout_image, cutout_label = random_crop(cutout_image,
                                                     cutout_label,
                                                     cutout_class,
                                                     1e-1,
                                                     0.3)
            cutout_label = _relocate_labels(cutout_label, cutout_box, image.size)
            cutout_labels.append(cutout_label)
            cutout_box = np.array(transpose_xy(convert_to_corners(unscale_boxes(cutout_box, image.size))), dtype=np.uint16)
            image_matrix = np.array(image)
            cutout_image = cutout_image.resize([cutout_box[3]-cutout_box[1], cutout_box[2]-cutout_box[0]])
            image_matrix[cutout_box[0]: cutout_box[2], cutout_box[1]: cutout_box[3]] = np.array(cutout_image)
            image = Image.fromarray(image_matrix)
        cutout_labels = np.concatenate(cutout_labels)
        return image, np.concatenate([labels, cutout_labels])

    image, labels = _cutmix_new_targets(image, labels, image_pool)
    return image, labels


def get_data_statistic(src_image_path, src_label_path, per_image=True):
    data = get_path_for_each_class(src_image_path, src_label_path, per_image)
    classes = ['not wear', 'wear', 'incorrect wear']
    plt.bar(classes, [data[0].shape[0], data[1].shape[0], data[2].shape[0]])
    plt.show()


def get_path_for_each_class(src_image_path, src_label_path, per_image=True, verbose=True):
    w_glob = []
    nw_glob = []
    iw_glob = []
    if verbose:
        loop = tqdm(zip(src_image_path, src_label_path),
                                       total=len(src_label_path),
                                       desc='Splitting')
    else:
        loop = zip(src_image_path, src_label_path)
    for image_path, label_path in loop:
        label = pd.read_csv(label_path, header=None, sep=' ').to_numpy()
        classes = label[..., 0]
        for cls in classes:
            if cls == 0:
                nw_glob.append([image_path, label_path])
            elif cls == 1:
                w_glob.append([image_path, label_path])
            else:
                iw_glob.append([image_path, label_path])
    if per_image:
        return np.unique(nw_glob, axis=0), np.unique(w_glob, axis=0), np.unique(iw_glob, axis=0)
    return np.unique(nw_glob, axis=0), np.unique(w_glob, axis=0), np.unique(iw_glob, axis=0)
