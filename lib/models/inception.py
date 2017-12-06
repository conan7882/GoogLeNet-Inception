#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: inception.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
from tensorflow.contrib.framework import add_arg_scope

from tensorcv.models.layers import *

# PATH = '/Users/gq/workspace/Dataset/pretrained/googlenet.npy'

# data_dict = np.load(PATH, encoding='latin1').item()
# print(data_dict)
# for key in data_dict:
#     print(key)
#     # for subkey in data_dict[key]:
#     #     print(subkey)


@add_arg_scope
def inception_layer(inputs,
                    conv_11_size,
                    conv_33_reduce_size, conv_33_size,
                    conv_55_reduce_size, conv_55_size,
                    pool_size,
                    data_dict={},
                    trainable=False,
                    name='inception'):

    arg_scope = tf.contrib.framework.arg_scope
    with arg_scope([conv], nl=tf.nn.relu, trainable=trainable,
                   data_dict=data_dict):
        conv_11 = conv(inputs, 1, conv_11_size, '{}_1x1'.format(name))

        conv_33_reduce = conv(inputs, 1, conv_33_reduce_size,
                              '{}_3x3_reduce'.format(name))
        conv_33 = conv(conv_33_reduce, 3, conv_33_size, '{}_3x3'.format(name))

        conv_55_reduce = conv(inputs, 1, conv_55_reduce_size,
                              '{}_5x5_reduce'.format(name))
        conv_55 = conv(conv_55_reduce, 5, conv_55_size, '{}_5x5'.format(name))

        pool = max_pool(inputs, '{}_pool'.format(name), stride=1, padding='SAME', filter_size=3)
        convpool = conv(pool, 1, pool_size, '{}_pool_proj'.format(name))

    return tf.concat([conv_11, conv_33, conv_55, convpool], 3, name='{}_concat'.format(name))
