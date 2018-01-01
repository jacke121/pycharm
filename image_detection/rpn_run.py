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

from image_detection.configs.cococonfig import CocoConfig
from image_detection.dataprovider import anchorgen, ssdgen
from image_detection.dataprovider.coco import CocoDataset
from image_detection.dataprovider.datagen import data_generator
from image_detection.models import rpn_model
from image_detection.utils import utils
import keras
import matplotlib.pyplot as plt

ROOT_DIR = os.getcwd() + "/output_rpn"
checkpoint_dir = os.path.join(ROOT_DIR, "checkpoints")
log_dir = os.path.join(ROOT_DIR, "logs")

mode = "training"
# mode = "inference"
dataset = "/home/andrew/PycharmProjects/algorithm-project/datasets/image/coco"
year = "2017"
save_period = 10
config = CocoConfig()
config.IMAGES_PER_GPU = 2
config.STEPS_PER_EPOCH = 1000
config.display()

if mode == "training":
    dataset_train = CocoDataset()
    dataset_train.load_coco(dataset, "train", year=year, class_ids=[2, 3])
    dataset_train.prepare()
    print(len(dataset_train.image_info))
    train_generator = ssdgen.data_generator(dataset_train, config, shuffle=True,
                                            batch_size=config.BATCH_SIZE)
else:
    config.RPN_NMS_THRESHOLD = 0.3

# Validation dataset
dataset_val = CocoDataset()
dataset_val.load_coco(dataset, "val", year=year, class_ids=[2, 3])
dataset_val.prepare()
print(len(dataset_val.image_info))
val_generator = ssdgen.data_generator(dataset_val, config, shuffle=True,
                                      batch_size=config.BATCH_SIZE,
                                      augment=False)

# Anchors
# [anchor_count, (y1, x1, y2, x2)]
anchors = anchorgen.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             config.BACKBONE_SHAPES,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

# Callbacks
callbacks = [
    keras.callbacks.TensorBoard(log_dir=log_dir,
                                histogram_freq=0, write_graph=True, write_images=False),
    keras.callbacks.ModelCheckpoint(checkpoint_dir + "/ck-{val_loss:.2f}.h5py",
                                    verbose=0, save_weights_only=True, period=save_period),
]

# Train
if os.name is 'nt':
    workers = 0
else:
    workers = max(config.BATCH_SIZE // 2, 2)

model = rpn_model.build_model(mode, config, anchors=anchors)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Load pre-trained weights
dirs = sorted(os.listdir(checkpoint_dir), cmp=lambda x, y: utils.cmp_time(x, y, checkpoint_dir))
if len(dirs) > 0:
    print("Loading the latest weights from {0}".format(dirs[0]))
    model.load_weights(os.path.join(checkpoint_dir, dirs[0]))

if mode == "training":
    model.fit_generator(
        train_generator,
        config.STEPS_PER_EPOCH,
        initial_epoch=0,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        validation_data=next(val_generator),
        validation_steps=config.VALIDATION_STEPS,
        max_queue_size=100,
        workers=workers,
        use_multiprocessing=True,
    )
else:
    # model.predict()
    for _ in range(5):
        inputs, outputs = next(val_generator)
        batch_images, batch_rpn_match, batch_rpn_bbox = inputs
        batch_images = batch_images.astype(np.float32) + config.MEAN_PIXEL
        print(np.max(batch_images))
        print(np.min(batch_images))
        print(batch_images.shape)
        print(batch_rpn_match.shape)
        print(batch_rpn_bbox.shape)
        preds, scores = model.predict(batch_images)

        height, width = config.IMAGE_SHAPE[:2]
        batch_pred_box = preds * np.array([[height, width, height, width]])
        print(batch_pred_box.shape)
        print(scores.shape)

        for i in range(len(batch_pred_box)):
            image = batch_images[i]
            pred_box = batch_pred_box[i]
            utils.draw_rpn(image.astype(np.uint8), pred_box, limit=10)
            print(scores[i, 1:10])
            plt.show()
