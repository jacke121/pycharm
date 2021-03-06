#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-12-24
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
import keras.layers as KL
import keras.models as KM
import tensorflow as tf
import numpy as np

from image_detection.layers.ProposalLayer import ProposalLayer
from image_detection.layers.RPN import build_rpn_model
from image_detection.layers.ResnetLayer import resnet_graph
from image_detection.loss import rpn_class_loss_graph, rpn_bbox_loss_graph


def build_model(mode, config, anchors=None):
    """Build Mask R-CNN architecture.
        input_shape: The shape of the input image.
        mode: Either "training" or "inference". The inputs and
            outputs of the model differ accordingly.
    """
    assert mode in ['training', 'inference']
    if mode == 'inference':
        assert anchors is not None

    # Image size must be dividable by 2 multiple times
    h, w = config.IMAGE_SHAPE[:2]
    if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")

    # Inputs
    input_image = KL.Input(
        shape=config.IMAGE_SHAPE.tolist(), name="input_image")

    # Build the shared convolutional layers.
    # Bottom-up Layers
    # Returns a list of the last layers of each stage, 5 in total.
    # Don't create the thead (stage 5), so we pick the 4th item in the list.
    _, C2, C3, C4, C5 = resnet_graph(input_image, "resnet101", stage5=True)
    # Top-down Layers
    # TODO: add assert to varify feature map sizes match what's in config
    P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
    P4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])
    P3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
    P2 = KL.Add(name="fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

    # Note that P6 is used in RPN, but not in the classifier heads.
    rpn_feature_maps = [P2, P3, P4, P5, P6]

    # RPN Model
    rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                          len(config.RPN_ANCHOR_RATIOS), 256)
    # Loop through pyramid layers
    layer_outputs = []  # list of lists
    for p in rpn_feature_maps:
        layer_outputs.append(rpn([p]))
    # Concatenate layer outputs
    # Convert from list of lists of level outputs to list of lists
    # of outputs across levels.
    # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
    output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
    outputs = list(zip(*layer_outputs))
    outputs = [KL.Concatenate(axis=1, name=n)(list(o))
               for o, n in zip(outputs, output_names)]

    rpn_class_logits, rpn_class, rpn_bbox = outputs

    if mode == "training":
        # RPN GT
        input_rpn_match = KL.Input(
            shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        input_rpn_bbox = KL.Input(
            shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

        # Losses
        rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
            [input_rpn_match, rpn_class_logits])
        rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
            [input_rpn_bbox, input_rpn_match, rpn_bbox])
        # Model
        inputs = [input_image, input_rpn_match, input_rpn_bbox]
        outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                   rpn_class_loss, rpn_bbox_loss]
        model = KM.Model(inputs, outputs, name='ssd')

        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=config.LEARNING_RATE, momentum=config.LEARNING_MOMENTUM,
                                         clipnorm=5.0)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        model._losses = []
        model._per_input_losses = {}
        loss_names = ["rpn_class_loss", "rpn_bbox_loss"]
        for name in loss_names:
            layer = model.get_layer(name)
            if layer.output in model.losses:
                continue
            model.add_loss(
                tf.reduce_mean(layer.output, keep_dims=True))

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [keras.regularizers.l2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                      for w in model.trainable_weights
                      if 'gamma' not in w.name and 'beta' not in w.name]
        model.add_loss(tf.add_n(reg_losses))

        # Compile
        model.compile(optimizer=optimizer, loss=[None] * len(model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in model.metrics_names:
                continue
            layer = model.get_layer(name)
            model.metrics_names.append(name)
            model.metrics_tensors.append(tf.reduce_mean(
                layer.output, keep_dims=True))

        return model

    else:
        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING
        rpn_rois, scores = ProposalLayer(proposal_count=proposal_count,
                                         nms_threshold=config.RPN_NMS_THRESHOLD,
                                         name="ROI",
                                         anchors=anchors,
                                         config=config)([rpn_class, rpn_bbox])

        model = KM.Model(input_image, [rpn_rois, scores], name='rpn')
        return model
