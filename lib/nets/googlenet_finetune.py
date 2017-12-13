#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: googlenet_finetune.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from tensorcv.models.layers import global_avg_pool, batch_norm, fc, dropout

from models.fine_tuning import Classification_Finetuning
from nets.googlenet import GoogleNet


class GoogLeNet_Finetuning(Classification_Finetuning):
    def get_grads(self):
        try:
            return self.grads
        except AttributeError:
            optimizer = self.get_optimizer()
            loss = self.get_loss()
            self.grads = optimizer.compute_gradients(loss)
            # [tf.summary.histogram('gradient/' + var.name, grad, 
            #   collections = [self.default_collection]) for grad, var in self.grads]
        return self.grads

    def _create_model(self):
        input_im = self.model_input[0]
        keep_prob = self.model_input[1]

        net = GoogleNet(num_class=self.nclass,
                        num_channels=self.nchannel,
                        im_height=self.im_height,
                        im_width=self.im_width,
                        is_load=self._is_load,
                        pre_train_path=self._pre_train_path,
                        is_rescale=False,
                        trainable=False)

        net.create_model([input_im, keep_prob])
        conv_out = net.layer['conv_out']

        # gap = global_avg_pool(conv_out)
        # gap_dropout = dropout(gap, keep_prob, self.is_training)

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([fc], trainable=True, wd=5e-4):
            fc6 = fc(conv_out, 1024, 'fc6')
            fc6_bn = batch_norm(fc6, train=self.is_training, name='fc6_bn')
            fc6_act = tf.nn.relu(fc6_bn)
            dropout_fc6 = dropout(fc6_act, keep_prob, self.is_training)

            # fc7 = fc(dropout_fc6, 2048, 'fc7')
            # fc7_bn = batch_norm(fc7, train=self.is_training, name='fc7_bn')
            # fc7_act = tf.nn.relu(fc7_bn)
            # dropout_fc7 = dropout(fc7_act, keep_prob, self.is_training)

            # fc7 = fc(dropout_fc6, 4096, 'fc7', nl=tf.nn.relu)
            # dropout_fc7 = dropout(fc7, keep_prob, self.is_training)

            fc8 = fc(dropout_fc6, self.nclass, 'fc8')

            # self.layer['fc6'] = fc6
            # self.layer['fc7'] = fc7
            self.layer['fc8'] = self.layer['output'] = fc8
            self.layer['class_prob'] = tf.nn.softmax(fc8, name='class_prob')
            self.layer['pre_prob'] = tf.reduce_max(self.layer['class_prob'],
                                                   axis=-1, name='pre_prob')
