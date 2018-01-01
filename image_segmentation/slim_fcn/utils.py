#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-12-23
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import tensorflow as tf
import numpy as np


def colormap(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1 << (7 - j)) * ((i & (1 << (3 * j))) >> (3 * j))
            g = g + (1 << (7 - j)) * ((i & (1 << (3 * j + 1))) >> (3 * j + 1))
            b = b + (1 << (7 - j)) * ((i & (1 << (3 * j + 2))) >> (3 * j + 2))

        cmap[i, :] = np.array([r, g, b])

    return cmap


class Colorize(object):
    def __init__(self):
        self.cmap = colormap(256)

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((size[0], size[1], 3))
        for i in range(color_image.shape[0]):
            for j in range(color_image.shape[1]):
                color_image[i, j, :] = self.cmap[gray_image[i, j]]

        return color_image


def colorize(ori_img):
    color_fcn = Colorize()
    img = color_fcn(ori_img.astype(np.uint8))
    return img
