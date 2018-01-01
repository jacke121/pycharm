#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-12-23
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from keras.preprocessing.image import Iterator
from keras.utils import OrderedEnqueuer, GeneratorEnqueuer


class SegDataIterator(Iterator):
    """
    支持voc和coco图像分割
    """

    def __init__(self, dataset, image_size=None, batch_size=32, shuffle=False, seed=None,
                 use_multiprocessing=True, workers=4,
                 max_queue_size=10, deprocess_X=None, deprocess_Y=None):
        super(SegDataIterator, self).__init__(len(dataset), batch_size, shuffle, seed)

        self.dataset = dataset
        self.image_size = image_size[1], image_size[0]
        self.enqueuer = GeneratorEnqueuer(self, use_multiprocessing=use_multiprocessing)
        self.workers = workers
        self.max_queue_size = max_queue_size
        self.is_start = False

        self.deprocess_X = deprocess_X
        self.deprocess_Y = deprocess_Y

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array), self.image_size[1], self.image_size[0], 3), np.float32)
        batch_y = np.zeros((len(index_array), self.image_size[1], self.image_size[0], self.dataset.num_classes),
                           np.float32)

        for i, j in enumerate(index_array):
            x = self.dataset.load_image(j, self.image_size)
            y = self.dataset.load_segmentation(j, self.image_size)
            batch_x[i] = x
            batch_y[i] = y
        if self.deprocess_X:
            batch_x = self.deprocess_X(batch_x)

        if self.deprocess_Y:
            batch_y = self.deprocess_Y(batch_y)

        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def data_gen(self):
        if not self.is_start:
            self.enqueuer.start(workers=self.workers, max_queue_size=self.max_queue_size)
            self.output_generator = self.enqueuer.get()
            self.is_start = True
        batch_x, batch_y = next(self.output_generator)
        return batch_x, batch_y
