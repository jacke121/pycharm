#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-12-24
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import tensorflow as tf
import numpy as np


def processing(x):
    return 2 * x / 255. - 1


def deprocessing(x):
    return (x + 1) / 2.


def vgg_processing(x):
    # RGB-mean
    x[:, :, :, 0] = x[:, :, :, 0] - 123.680
    x[:, :, :, 1] = x[:, , :, 1] - 116.779
    x[:, :, :, 2] = x[:, , :, 2] - 103.939


def vgg_deprocessing(x):
    # RGB-mean
    x[:, :, :, 0] = x[:, :, :, 0] + 123.680
    x[:, :, :, 1] = x[:, :, :, 1] + 116.779
    x[:, :, :, 2] = x[:, :, :, 2] + 103.939
