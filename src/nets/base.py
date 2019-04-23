#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
from abc import abstractmethod

import src.utils.multigpu as multigpu


class BaseModel(object):
    """ Base model """

    def set_is_training(self, is_training=True):
        self.is_training = is_training

    def get_loss(self):
        try:
            return self._loss
        except AttributeError:
            self._loss = self._get_loss()
        return self._loss

    def _get_loss(self):
        raise NotImplementedError()

    def get_optimizer(self):
        try:
            return self._opt
        except AttributeError:
            self._opt = self._get_optimizer()
        return self._opt

    def _get_optimizer(self):
        raise NotImplementedError()

    def get_train_op(self, moniter=False):
        tower_grads = []
        opt = self.get_optimizer()

        for i in range(2):
            with tf.device('/gpu:{}'.format(i)) as scope:
                with tf.name_scope('train_{}'.format(i)):
                    loss = self.get_loss()
                    var_list = tf.trainable_variables()
                    grads = tf.gradients(loss, var_list)
                    # if moniter:
                    #     [tf.summary.histogram('generator_gradient/' + var.name, grad, 
                    #         collections=['train']) for grad, var in zip(grads, var_list)]
                    tower_grads.append(zip(grads, var_list))
        grads = multigpu.average_gradients(tower_grads)     
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(grads)
        return train_op

    
