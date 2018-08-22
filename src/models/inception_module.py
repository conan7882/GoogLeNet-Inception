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
def inception_layer(conv_11_size, conv_33_reduce_size, conv_33_size,
                    conv_55_reduce_size, conv_55_size, pool_size,
                    layer_dict, inputs=None,
                    bn=False, wd=0, init_w=None,
                    pretrained_dict=None, trainable=True, is_training=True,
                    name='inception'):
    
    if inputs is None:
        inputs = layer_dict['cur_input']
    layer_dict['cur_input'] = inputs

    arg_scope = tf.contrib.framework.arg_scope
    with arg_scope([L.conv], layer_dict=layer_dict, pretrained_dict=pretrained_dict,
                   bn=bn, nl=tf.nn.relu, init_w=init_w, trainable=trainable,
                   is_training=is_training, wd=wd, add_summary=False):


        conv_11 = L.conv(filter_size=1, out_dim=conv_11_size,
                         inputs=inputs, name='{}_1x1'.format(name))

        L.conv(filter_size=1, out_dim=conv_33_reduce_size,
               inputs=inputs, name='{}_3x3_reduce'.format(name))
        conv_33 = L.conv(filter_size=3, out_dim=conv_33_size,
                         name='{}_3x3'.format(name))

        L.conv(filter_size=1, out_dim=conv_55_reduce_size,
               inputs=inputs, name='{}_5x5_reduce'.format(name))
        conv_55 = L.conv(filter_size=5, out_dim=conv_55_size,
                         name='{}_5x5'.format(name))

        L.max_pool(layer_dict=layer_dict, inputs=inputs, stride=1,
                   filter_size=3, padding='SAME', name='{}_pool'.format(name))
        convpool = L.conv(filter_size=1, out_dim=pool_size,
                          name='{}_pool_proj'.format(name))

        # conv_11 = conv(inputs, 1, conv_11_size, '{}_1x1'.format(name))

        # conv_33_reduce = conv(inputs, 1, conv_33_reduce_size,
        #                       '{}_3x3_reduce'.format(name))
        # conv_33 = conv(conv_33_reduce, 3, conv_33_size, '{}_3x3'.format(name))

        # conv_55_reduce = conv(inputs, 1, conv_55_reduce_size,
        #                       '{}_5x5_reduce'.format(name))
        # conv_55 = conv(conv_55_reduce, 5, conv_55_size, '{}_5x5'.format(name))

        # pool = max_pool(inputs, '{}_pool'.format(name), stride=1,
        #                 padding='SAME', filter_size=3)
        # convpool = conv(pool, 1, pool_size, '{}_pool_proj'.format(name))
        output = tf.concat([conv_11, conv_33, conv_55, convpool], 3,
                           name='{}_concat'.format(name))
        layer_dict['cur_input'] = output
        layer_dict[name] = output
    return output

def inception_conv_layers(layer_dict, inputs=None, pretrained_dict=None,
                          bn=False, wd=0, init_w=None,
                          is_training=True, trainable=True):
    if inputs is None:
        inputs = layer_dict['cur_input']
    layer_dict['cur_input'] = inputs
    
    arg_scope = tf.contrib.framework.arg_scope
    with arg_scope([L.conv], layer_dict=layer_dict, pretrained_dict=pretrained_dict,
                   bn=bn, nl=tf.nn.relu, init_w=init_w, trainable=trainable,
                   is_training=is_training, wd=wd, add_summary=False):

        conv1 = L.conv(7, 64, inputs=inputs, name='conv1_7x7_s2', stride=2)
        padding1 = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        conv1_pad = tf.pad(conv1, padding1, 'CONSTANT')
        pool1, _ = L.max_pool(
            layer_dict=layer_dict, inputs=conv1_pad, stride=2,
            filter_size=3, padding='VALID', name='pool1')
        pool1_lrn = tf.nn.local_response_normalization(
            pool1, depth_radius=2, alpha=2e-05, beta=0.75,
            name='pool1_lrn')

        conv2_reduce = L.conv(1, 64, inputs=pool1_lrn, name='conv2_3x3_reduce')
        conv2 = L.conv(3, 192, inputs=conv2_reduce, name='conv2_3x3')
        padding2 = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        conv2_pad = tf.pad(conv2, padding2, 'CONSTANT')
        pool2, _ = L.max_pool(
            layer_dict=layer_dict, inputs=conv2_pad, stride=2,
            filter_size=3, padding='VALID', name='pool2')
        pool2_lrn = tf.nn.local_response_normalization(
            pool2, depth_radius=2, alpha=2e-05, beta=0.75,
            name='pool2_lrn')
    layer_dict['cur_input'] = pool2_lrn
    return pool2_lrn

def inception_layers(layer_dict, inputs=None, pretrained_dict=None,
                     bn=False, init_w=None, wd=0,
                     trainable=True, is_training=True):
    if inputs is not None:
        layer_dict['cur_input'] = inputs

    arg_scope = tf.contrib.framework.arg_scope
    with arg_scope([inception_layer], layer_dict=layer_dict,
                   pretrained_dict=pretrained_dict,
                   bn=bn, init_w=init_w, trainable=trainable,
                   is_training=is_training, wd=wd):

        inception_layer(64, 96, 128, 16, 32, 32, name='inception_3a')
        inception_layer(128, 128, 192, 32, 96, 64, name='inception_3b')
        L.max_pool(layer_dict, stride=2, filter_size=3, name='pool3')

        inception_layer(192, 96, 208, 16, 48, 64, name='inception_4a')
        inception_layer(160, 112, 224, 24, 64, 64, name='inception_4b')
        inception_layer(128, 128, 256, 24, 64, 64, name='inception_4c')
        inception_layer(112, 144, 288, 32, 64, 64, name='inception_4d')
        inception_layer(256, 160, 320, 32, 128, 128, name='inception_4e')
        L.max_pool(layer_dict, stride=2, filter_size=3, name='pool4')

        inception_layer(256, 160, 320, 32, 128, 128, name='inception_5a')
        inception_layer(384, 192, 384, 48, 128, 128, name='inception_5b')

    return layer_dict['cur_input']

def inception_fc(layer_dict, n_class, keep_prob, inputs=None,
                 pretrained_dict=None, is_training=True,
                 bn=False, init_w=None, trainable=True, wd=0):

    if inputs is not None:
        layer_dict['cur_input'] = inputs

    layer_dict['cur_input'] = L.global_avg_pool(layer_dict['cur_input'], keepdims=True)
    # layer_dict['cur_input'] = tf.expand_dims(layer_dict['cur_input'], [1, 2])
    L.drop_out(layer_dict, is_training, keep_prob=keep_prob)
    L.conv(filter_size=1, out_dim=n_class, layer_dict=layer_dict,
           pretrained_dict=pretrained_dict, trainable=trainable,
           bn=bn, init_w=init_w, wd=wd, is_training=is_training,
           name='loss3_classifier')
    layer_dict['cur_input'] = tf.squeeze(layer_dict['cur_input'], [1, 2])

    return layer_dict['cur_input']


    # with arg_scope([inception_layer],
    #                    trainable=self._trainable,
    #                    data_dict=data_dict):
    #         # inception3a = inception_layer(
    #         #     pool2_lrn, 64, 96, 128, 16, 32, 32, name='inception_3a')
    #         # inception3b = inception_layer(
    #         #     inception3a, 128, 128, 192, 32, 96, 64, name='inception_3b')
    #         # pool3 = max_pool(
    #         #     inception3b, 'pool3', padding='SAME', filter_size=3, stride=2)

    #         # inception4a = inception_layer(
    #         #     pool3, 192, 96, 208, 16, 48, 64, name='inception_4a')
    #         # inception4b = inception_layer(
    #         #     inception4a, 160, 112, 224, 24, 64, 64, name='inception_4b')
    #         # inception4c = inception_layer(
    #         #     inception4b, 128, 128, 256, 24, 64, 64, name='inception_4c')
    #         # inception4d = inception_layer(
    #         #     inception4c, 112, 144, 288, 32, 64, 64, name='inception_4d')
    #         # inception4e = inception_layer(
    #         #     inception4d, 256, 160, 320, 32, 128, 128, name='inception_4e')
    #         # pool4 = max_pool(
    #         #     inception4e, 'pool4', padding='SAME', filter_size=3, stride=2)

    #         inception5a = inception_layer(
    #             pool4, 256, 160, 320, 32, 128, 128, name='inception_5a')
    #         inception5b = inception_layer(
    #             inception5a, 384, 192, 384, 48, 128, 128, name='inception_5b')





