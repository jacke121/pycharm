#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-11-26
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import sys
import os
import matplotlib.pyplot as plt

import collections
import os.path as osp

import numpy as np
import PIL.Image
import glob
import tensorflow as tf
import math

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'


class VOCClassSegBase(object):
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
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, dataset_dir, split='train', size=(256, 256), shuffle=False, transform=True):
        self.root = root
        self.split = split
        self.size = size
        if transform:
            self._transform = self.transform
        else:
            self._transform = transform
        self.shuffle = shuffle

        # VOC2012
        self.files = collections.defaultdict(list)
        for split in ['train', 'val', 'trainval']:
            imgsets_file = osp.join(
                dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                lbl_file = osp.join(
                    dataset_dir, 'SegmentationClass/%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = img.resize(self.size)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = lbl.resize(self.size)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def data_generate(self):
        indexs = [i for i in range(len(self))]
        if self.shuffle:
            random.shuffle(indexs)
        for index in indexs:
            yield self[index]

    def data_write_pair(self, output_dir):
        assert output_dir and not os.path.exists(output_dir)
        os.mkdir(output_dir)
        for index in range(len(self)):
            data_file = self.files[self.split][index]
            img_file = data_file['img']
            print(index)
            print(os.path.basename(img_file))
            img = plt.imread(img_file)
            img = img / 255.
            lbl_file = data_file['lbl']
            lbl = plt.imread(lbl_file)
            total_img = np.concatenate((img, lbl), axis=1)
            # print(total_img.shape)
            output_path = os.path.join(output_dir, os.path.basename(img_file))
            im = PIL.Image.fromarray(np.uint8(total_img*255))
            im.save(output_path)
            # abc = plt.imread(output_path)
            # print(abc.shape)
            # exit()

    def transform(self, img, lbl):
        img = img / 255.
        lbl[lbl < 0] = 0
        return img, lbl

    def untransform(self, img, lbl):
        img = img * 255.
        return img, lbl

        # def transform(self, img, lbl):
        #     img = img[:, :, ::-1]  # RGB -> BGR
        #     img = img.astype(np.float64)
        #     img -= self.mean_bgr
        #     img = img.transpose(2, 0, 1)
        #     return img, lbl
        #
        # def untransform(self, img, lbl):
        #     img = img.numpy()
        #     img = img.transpose(1, 2, 0)
        #     img += self.mean_bgr
        #     img = img.astype(np.uint8)
        #     img = img[:, :, ::-1]
        #     lbl = lbl.numpy()
        #     return img, lbl


if __name__ == '__main__':
    mode = "train"
    input_dir = os.path.join(root, "datasets/image/VOC2012")
    output_dir = os.path.join(root, "datasets/image/voc_pair2012", mode)

    voc = VOCClassSegBase("/home/andrew/PycharmProjects/algorithm-project/datasets/image/VOC2012", split="trainval")
    print(len(voc))
    voc.data_write_pair(output_dir)
