#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: generator.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf

class Generator(object):
    def __init__(self, data_generator, output_types, output_shapes, batch_size,
                 buffer_size, num_parallel_preprocess):

        with tf.name_scope('data_generator'):
            dataset = tf.data.Dataset.from_generator(
                data_generator,
                output_types=output_types,
                output_shapes=output_shapes,
                )

            self._n_preprocess = num_parallel_preprocess
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(buffer_size=buffer_size)
            dataset = dataset.shuffle(buffer_size=buffer_size * 10)

            self.iter = dataset.make_initializable_iterator()
            self.batch_data = self.iter.get_next()
            self.dataset = dataset

    def init_iterator(self, sess, reset_scale=False):
        sess.run(self.iter.initializer)
