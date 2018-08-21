#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vgg.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf

from src.nets.base import BaseModel
import src.models.layers as L
import src.models.vgg_module as module


INIT_W = tf.keras.initializers.he_normal()

class BaseVGG(BaseModel):
    """ base model of VGG for image classification """

    def __init__(self, n_channel, n_class, pre_trained_path=None,
                 bn=False, wd=0, trainable=True, sub_vgg_mean=True):
        self._n_channel = n_channel
        self.n_class = n_class
        self._bn = bn
        self._wd = wd
        self._trainable = trainable
        self._sub_vgg_mean = sub_vgg_mean

        self._pretrained_dict = None
        if pre_trained_path:
            self._pretrained_dict = np.load(
                pre_trained_path, encoding='latin1').item()

        self.layers = {}

    def _create_train_input(self):
        self.image = tf.placeholder(
            tf.float32, [None, None, None, self._n_channel], name='image')
        self.label = tf.placeholder(tf.int64, [None], 'label')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.lr = tf.placeholder(tf.float32, name='lr')

    def _create_test_input(self):
        self.image = tf.placeholder(
            tf.float32, [None, None, None, self._n_channel], name='image')
        self.label = tf.placeholder(tf.int64, [None], 'label')
        self.keep_prob = 1.

    def create_train_model(self):
        self.set_is_training(is_training=True)
        self._create_train_input()
        if self._sub_vgg_mean:
            vgg_input = module.sub_rgb2bgr_mean(self.image)
        else:
            vgg_input = self.image

        with tf.variable_scope('conv_layers', reuse=tf.AUTO_REUSE):
            self.layers['conv_out'] = self._conv_layers(vgg_input)
        with tf.variable_scope('fc_layers', reuse=tf.AUTO_REUSE):   
            self.layers['logits'] = self._fc_layers(self.layers['conv_out'])
            self.layers['gap_out'] = L.global_avg_pool(self.layers['logits'])

    def create_test_model(self):
        self.set_is_training(is_training=False)
        self._create_test_input()
        if self._sub_vgg_mean:
            vgg_input = module.sub_rgb2bgr_mean(self.image)
        else:
            vgg_input = self.image

        with tf.variable_scope('conv_layers', reuse=tf.AUTO_REUSE):
            self.layers['conv_out'] = self._conv_layers(vgg_input)
        with tf.variable_scope('fc_layers', reuse=tf.AUTO_REUSE):   
            self.layers['logits'] = self._fc_layers(self.layers['conv_out'])
            self.layers['gap_out'] = L.global_avg_pool(self.layers['logits'])
            self.layers['top_5'] = tf.nn.top_k(
                tf.nn.softmax(self.layers['gap_out']), k=5, sorted=True)

    def _conv_layers(self, inputs):
        raise NotImplementedError()

    def _fc_layers(self, inputs):
        fc_out = module.vgg_fc(
            layer_dict=self.layers, n_class=self.n_class, keep_prob=self.keep_prob,
            inputs=inputs, pretrained_dict=self._pretrained_dict,
            bn=self._bn, init_w=INIT_W, trainable=self._trainable,
            is_training=self.is_training, wd=self._wd)
        return fc_out

    def _get_loss(self):
        with tf.name_scope('loss'):
            labels = self.label
            logits = self.layers['gap_out']
            # logits = tf.squeeze(logits, axis=1)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits,
                name='cross_entropy')
            return tf.reduce_mean(cross_entropy)

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(self.lr)

    def get_accuracy(self):
        with tf.name_scope('accuracy'):
            prediction = tf.argmax(self.layers['gap_out'], axis=1)
            correct_prediction = tf.equal(prediction, self.label)
            return tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), 
                name = 'result')

class VGG19(BaseVGG):
   def _conv_layers(self, inputs):
        conv_out = module.vgg19_conv(
            layer_dict=self.layers, inputs=inputs, keep_prob=1.,
            pretrained_dict=self._pretrained_dict,
            bn=self._bn, init_w=INIT_W, trainable=self._trainable,
            is_training=self.is_training, wd=self._wd)
        return conv_out

class VGG16(BaseVGG):
   def _conv_layers(self, inputs):
        conv_out = module.vgg16_conv(
            layer_dict=self.layers, inputs=inputs, keep_prob=1.,
            pretrained_dict=self._pretrained_dict,
            bn=self._bn, init_w=INIT_W, trainable=self._trainable,
            is_training=self.is_training, wd=self._wd)
        return conv_out

class VGG_CIFAR10(VGG19):
    def _fc_layers(self, inputs):
        fc_out = module.vgg_small_fc(
            layer_dict=self.layers, n_class=self.n_class, keep_prob=self.keep_prob,
            inputs=inputs, pretrained_dict=self._pretrained_dict,
            bn=self._bn, init_w=INIT_W, trainable=self._trainable,
            is_training=self.is_training, wd=self._wd)
        return fc_out
