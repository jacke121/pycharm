#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-12-23
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt

from dataprovider.coco import CocoDataset
from dataprovider.colormap import colorize
from dataprovider.dataloader import SegDataIterator
from dataprovider.voc import VOCDataset


def draw_image(dataset_train):
    for i in range(5):
        img = dataset_train.load_image(i, image_size=(224, 224))
        print(np.min(img))
        print(np.max(img))
        print(img.shape)
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(img.astype(np.uint8))

        mask = dataset_train.load_segmentation(i, image_size=(224, 224))
        mask = np.argmax(mask, axis=-1)
        print(np.min(mask))
        print(np.max(mask))
        print(mask.shape)
        mask = colorize(mask)
        plt.subplot(1, 3, 2)
        plt.imshow(mask)

        mask = dataset_train.load_segmentation(i)
        mask = np.argmax(mask, axis=-1)
        print(np.min(mask))
        print(np.max(mask))
        print(mask.shape)
        mask = colorize(mask)
        plt.subplot(1, 3, 3)
        plt.imshow(mask)

        plt.show()

    plt.show()
    exit()


def draw_image_batch(batch_x, batch_y):
    img = batch_x[1]
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img.astype(np.uint8))
    mask = batch_y[1]
    mask = np.argmax(mask, axis=-1)

    mask = colorize(mask)
    plt.subplot(1, 2, 3)
    plt.imshow(mask)


if __name__ == '__main__':
    dataset_dir = "/home/andrew/PycharmProjects/algorithm-project/datasets/image/coco"
    subset = "train"
    year = '2017'
    dataset_train = CocoDataset()
    dataset_train.load_coco(dataset_dir, subset, year=year)

    # subset = "trainval"
    # year = "2012"
    # dataset_dir = "/home/andrew/PycharmProjects/algorithm-project/datasets/image/VOC"
    # dataset_train = VOCDataset(dataset_dir, year=year,split=subset)

    print(len(dataset_train))
    print(dataset_train.get_classes())
    print(len(dataset_train.get_classes()))

    # draw_image(dataset_train)

    segDataIterator = SegDataIterator(dataset_train, image_size=(224, 224), batch_size=32, shuffle=False)

    for i in range(5):
        x, y = next(segDataIterator)
        # x, y = segDataIterator.data_gen()
        draw_image_batch(x, y)

    plt.show()
