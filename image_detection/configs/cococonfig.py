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

from image_detection.configs.config import Config


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 80  # COCO has 80 classes
    NUM_CLASSES = 1 + 90  # COCO has 80 classes in 2017
