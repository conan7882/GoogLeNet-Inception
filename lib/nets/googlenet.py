#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: googlenet.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import numpy as np

from tensorcv.models.layers import conv, fc, global_avg_pool, dropout, max_pool
from tensorcv.models.base import BaseModel

from models.inception import inception_layer

MEAN = [103.939, 116.779, 123.68]


def resize_tensor_image_with_smallest_side(image, small_size):
    """
    Resize image tensor with smallest side = small_size and
    keep the original aspect ratio.

    Args:
        image (tf.tensor): 4-D Tensor of shape
            [batch, height, width, channels] or 3-D Tensor of shape
            [height, width, channels].
        small_size (int): A 1-D int. The smallest side of resize image.

    Returns:
        Image tensor with the original aspect ratio and
        smallest side = small_size .
        If images was 4-D, a 4-D float Tensor of shape
        [batch, new_height, new_width, channels].
        If images was 3-D, a 3-D float Tensor of shape
        [new_height, new_width, channels].
    """
    im_shape = tf.shape(image)
    shape_dim = image.get_shape()
    if len(shape_dim) <= 3:
        height = tf.cast(im_shape[0], tf.float32)
        width = tf.cast(im_shape[1], tf.float32)
    else:
        height = tf.cast(im_shape[1], tf.float32)
        width = tf.cast(im_shape[2], tf.float32)

    height_smaller_than_width = tf.less_equal(height, width)

    new_shorter_edge = tf.constant(small_size, tf.float32)
    new_height, new_width = tf.cond(
        height_smaller_than_width,
        lambda: (new_shorter_edge, (width / height) * new_shorter_edge),
        lambda: ((height / width) * new_shorter_edge, new_shorter_edge))

    return tf.image.resize_images(
        tf.cast(image, tf.float32),
        [tf.cast(new_height, tf.int32), tf.cast(new_width, tf.int32)])


class GoogleNet(BaseModel):
    def __init__(self, num_class=1000,
                 num_channels=3,
                 im_height=224, im_width=224,
                 learning_rate=0.0001,
                 is_load=False,
                 pre_train_path=None,
                 is_rescale=False,
                 trainable=False):

        self._lr = learning_rate
        self.num_channels = num_channels
        self.im_height = im_height
        self.im_width = im_width
        self.num_class = num_class
        self._is_rescale = is_rescale
        self._trainable = trainable

        self.layer = {}

        self._is_load = is_load
        if self._is_load and pre_train_path is None:
            raise ValueError('pre_train_path can not be None!')
        self._pre_train_path = pre_train_path

        self.set_is_training(True)

    def _create_input(self):
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(
            tf.float32, name='image',
            shape=[None, self.im_height, self.im_width, self.num_channels])

        self.label = tf.placeholder(tf.int64, [None], 'label')

        self.set_model_input([self.image, self.keep_prob])
        self.set_dropout(self.keep_prob, keep_prob=0.4)
        self.set_train_placeholder([self.image, self.label])
        self.set_prediction_placeholder(self.image)

    def _create_conv(self, inputs, data_dict):
        arg_scope = tf.contrib.framework.arg_scope

        with arg_scope([conv], trainable=self._trainable,
                       data_dict=data_dict, nl=tf.nn.relu):
            conv1 = conv(inputs, 7, 64, name='conv1_7x7_s2', stride=2)
            padding1 = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
            conv1_pad = tf.pad(conv1, padding1, 'CONSTANT')
            pool1 = max_pool(
                conv1_pad, 'pool1', padding='VALID', filter_size=3, stride=2)
            pool1_lrn = tf.nn.local_response_normalization(
                pool1, depth_radius=2, alpha=2e-05, beta=0.75,
                name='pool1_lrn')

            conv2_reduce = conv(pool1_lrn, 1, 64, name='conv2_3x3_reduce')
            conv2 = conv(conv2_reduce, 3, 192, name='conv2_3x3')
            padding2 = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
            conv2_pad = tf.pad(conv2, padding1, 'CONSTANT')
            pool2 = max_pool(
                conv2_pad, 'pool2', padding='VALID', filter_size=3, stride=2)
            pool2_lrn = tf.nn.local_response_normalization(
                pool2, depth_radius=2, alpha=2e-05, beta=0.75,
                name='pool2_lrn')

        with arg_scope([inception_layer],
                       trainable=self._trainable,
                       data_dict=data_dict):
            inception3a = inception_layer(
                pool2_lrn, 64, 96, 128, 16, 32, 32, name='inception_3a')
            inception3b = inception_layer(
                inception3a, 128, 128, 192, 32, 96, 64, name='inception_3b')
            pool3 = max_pool(
                inception3b, 'pool3', padding='SAME', filter_size=3, stride=2)

            inception4a = inception_layer(
                pool3, 192, 96, 208, 16, 48, 64, name='inception_4a')
            inception4b = inception_layer(
                inception4a, 160, 112, 224, 24, 64, 64, name='inception_4b')
            inception4c = inception_layer(
                inception4b, 128, 128, 256, 24, 64, 64, name='inception_4c')
            inception4d = inception_layer(
                inception4c, 112, 144, 288, 32, 64, 64, name='inception_4d')
            inception4e = inception_layer(
                inception4d, 256, 160, 320, 32, 128, 128, name='inception_4e')
            pool4 = max_pool(
                inception4e, 'pool4', padding='SAME', filter_size=3, stride=2)

            inception5a = inception_layer(
                pool4, 256, 160, 320, 32, 128, 128, name='inception_5a')
            inception5b = inception_layer(
                inception5a, 384, 192, 384, 48, 128, 128, name='inception_5b')

        return inception5b

    def _create_model(self):
        with tf.name_scope('input'):
            input_im = self.model_input[0]
            keep_prob = self.model_input[1]

            if self._is_rescale:
                input_im =\
                    resize_tensor_image_with_smallest_side(input_im, 224)
            self.layer['input'] = input_im

            red, green, blue = tf.split(axis=3, num_or_size_splits=3,
                                        value=input_im)

            input_bgr = tf.concat(axis=3, values=[
                blue - MEAN[0],
                green - MEAN[1],
                red - MEAN[2],
            ])

        data_dict = {}
        if self._is_load:
            data_dict = np.load(self._pre_train_path, encoding='latin1').item()

        inception5b = self._create_conv(input_bgr, data_dict)

        gap = global_avg_pool(inception5b)
        gap_dropout = dropout(gap, keep_prob, self.is_training)

        fc1 = fc(gap_dropout, 1000, 'loss3_classifier', data_dict=data_dict)

        self.layer['conv_out'] = inception5b
        self.layer['output'] = fc1
        self.layer['class_prob'] = tf.nn.softmax(fc1, name='class_prob')
        self.layer['pre_prob'] = tf.reduce_max(self.layer['class_prob'],
                                               axis=-1, name='pre_prob')
        self.layer['prediction'] = tf.argmax(self.layer['output'], axis=-1)
