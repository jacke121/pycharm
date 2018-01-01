"""Pascal VOC Segmenttion Generator."""
from __future__ import unicode_literals
import os

from PIL import Image
import numpy as np
import collections


class VOCDataset(object):
    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])

    def __init__(self, dataset_dir, year='2012', split='train'):
        self.split = split
        self.year = year
        self.dataset_dir = dataset_dir + year

        self.files = collections.defaultdict(list)
        for split in ['train', 'val', 'trainval']:
            imgsets_file = os.path.join(
                self.dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = os.path.join(self.dataset_dir, 'JPEGImages/%s.jpg' % did)
                lbl_file = os.path.join(
                    self.dataset_dir, 'SegmentationClass/%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })
        self.num_classes = len(VOCDataset.class_names)

    def __len__(self):
        return len(self.files[self.split])

    def get_classes(self):
        return list(VOCDataset.class_names)

    def load_image(self, image_id, image_size=None):
        data_file = self.files[self.split][image_id]
        img_file = data_file['img']
        img = Image.open(img_file)
        if image_size:
            img = img.resize(image_size)
        img = np.asarray(img, np.uint8)

        return img

    def load_segmentation(self, image_id, image_size=None):
        data_file = self.files[self.split][image_id]
        lbl_file = data_file['lbl']
        lbl = Image.open(lbl_file)
        if image_size:
            lbl = lbl.resize(image_size)
        lbl = np.asarray(lbl, np.int32)
        lbl[lbl == 255] = 0

        final_mask = np.zeros(lbl.shape + (self.num_classes,), dtype=np.uint8)
        for i in range(final_mask.shape[0]):
            for j in range(final_mask.shape[1]):
                final_mask[i, j, lbl[i, j]] = 1

        return final_mask
