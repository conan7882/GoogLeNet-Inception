#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: fine_tuning.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from tensorcv.models.layers import global_avg_pool, batch_norm, fc, dropout
from tensorcv.models.base import BaseModel


class Net_Finetuning(BaseModel):
    def __init__(self, num_class=1000,
                 num_channels=3,
                 im_height=None, im_width=None,
                 learning_rate=0.0001,
                 is_load=False,
                 pre_train_path=None,
                 drop_out=0.5):

        self._lr = learning_rate
        self.nchannel = num_channels
        self.im_height = im_height
        self.im_width = im_width
        self.nclass = num_class
        # self._is_rescale = is_rescale
        self._dropout = drop_out

        self.layer = {}

        self._is_load = is_load
        if self._is_load and pre_train_path is None:
            raise ValueError('pre_train_path can not be None!')
        self._pre_train_path = pre_train_path

        self.set_is_training(True)

    def _create_input(self):
        pass

    def _create_model(self):
        pass

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(beta1=0.9,
                                      learning_rate=self._lr)

    def _get_loss(self):
        pass

    def _ex_setup_graph(self):
        pass


class Classification_Finetuning(Net_Finetuning):

    def _create_input(self):
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(
            tf.float32, name='image',
            shape=[None, self.im_height, self.im_width, self.nchannel])

        self.label = tf.placeholder(tf.int64, [None], 'label')

        self.set_model_input([self.image, self.keep_prob])
        self.set_dropout(self.keep_prob, keep_prob=self._dropout)
        self.set_train_placeholder([self.image, self.label])
        self.set_prediction_placeholder([self.image, self.label])

    def _create_model(self):
        pass

    def _get_loss(self):
        with tf.name_scope('loss'):
            cross_entropy =\
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.label,
                    logits=self.layer['output'])
            cross_entropy_loss = tf.reduce_mean(
                cross_entropy, name='cross_entropy')
            tf.summary.scalar('cross_entropy_loss', cross_entropy_loss, collections=['train'])
            tf.add_to_collection('losses', cross_entropy_loss)
            return tf.add_n(tf.get_collection('losses'), name='result')

    def _ex_setup_graph(self):
        with tf.name_scope('accuracy'):
            prediction = tf.argmax(self.layer['output'], axis=-1)
            correct_prediction = tf.equal(prediction, self.label)
            self.accuracy = tf.reduce_mean(
                        tf.cast(correct_prediction, tf.float32),
                        name = 'result')
            tf.summary.scalar(
                'accuracy', self.accuracy, collections=['train'])
