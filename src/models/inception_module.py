#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: inception_module.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

import src.models.layers as L


def sub_rgb2bgr_mean(inputs):
    with tf.name_scope('sub_mean'):
        red, green, blue = tf.split(axis=3,
                                    num_or_size_splits=3,
                                    value=inputs)

        imagenet_mean = [103.939, 116.779, 123.68]

        input_bgr = tf.concat(axis=3, values=[
            blue - imagenet_mean[0],
            green - imagenet_mean[1],
            red - imagenet_mean[2],
        ])
        return input_bgr

@add_arg_scope
def inception_layer(
                    conv_11_size,
                    conv_33_reduce_size, conv_33_size,
                    conv_55_reduce_size, conv_55_size,
                    pool_size,
                    layer_dict,
                    inputs=None,
                    pretrained_dict=None,
                    trainable=True,
                    name='inception'):
    
    if inputs
    arg_scope = tf.contrib.framework.arg_scope
    with arg_scope([L.conv], layer_dict=layer_dict,

        nl=tf.nn.relu, trainable=trainable,
                   data_dict=data_dict):


conv(filter_size,
         out_dim,
         layer_dict,
         inputs=None,
         pretrained_dict=None,
         stride=1,
         dilations=[1, 1, 1, 1],
         bn=False,
         nl=tf.identity,
         init_w=None,
         init_b=tf.zeros_initializer(),
         use_bias=True,
         padding='SAME',
         pad_type='ZERO',
         trainable=True,
         is_training=None,
         wd=0,
         name='conv',
         add_summary=False)

        conv_11 = conv(inputs, 1, conv_11_size, '{}_1x1'.format(name))

        conv_33_reduce = conv(inputs, 1, conv_33_reduce_size,
                              '{}_3x3_reduce'.format(name))
        conv_33 = conv(conv_33_reduce, 3, conv_33_size, '{}_3x3'.format(name))

        conv_55_reduce = conv(inputs, 1, conv_55_reduce_size,
                              '{}_5x5_reduce'.format(name))
        conv_55 = conv(conv_55_reduce, 5, conv_55_size, '{}_5x5'.format(name))

        pool = max_pool(inputs, '{}_pool'.format(name), stride=1,
                        padding='SAME', filter_size=3)
        convpool = conv(pool, 1, pool_size, '{}_pool_proj'.format(name))

    return tf.concat([conv_11, conv_33, conv_55, convpool],
                     3, name='{}_concat'.format(name))