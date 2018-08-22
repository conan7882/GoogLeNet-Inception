#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: evaluator.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np
import tensorflow as tf


class Evaluator(object):
    def __init__(self, model):

        self._model = model
        self._accuracy_op = model.get_accuracy()

    def accuracy(self, sess, dataflow):
        dataflow.reset_epoch()

        step = 0
        acc_sum = 0
        while dataflow.epochs_completed < 1:
            step += 1

            batch_data = dataflow.next_batch_dict()
            im = batch_data['image']
            label = batch_data['label']
            acc = sess.run(
                self._accuracy_op, 
                feed_dict={self._model.image: im,
                           self._model.label: label})
            acc_sum += acc
        print('[accuracy]: {:.04f}'.format(acc_sum / step))
