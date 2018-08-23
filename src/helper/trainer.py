#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: trainer.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np
import tensorflow as tf


def display(global_step,
            step,
            scaler_sum_list,
            name_list,
            collection,
            summary_val=None,
            summary_writer=None,
            ):
    print('[step: {}]'.format(global_step), end='')
    for val, name in zip(scaler_sum_list, name_list):
        print(' {}: {:.4f}'.format(name, val * 1. / step), end='')
    print('')
    if summary_writer is not None:
        s = tf.Summary()
        for val, name in zip(scaler_sum_list, name_list):
            s.value.add(tag='{}/{}'.format(collection, name),
                        simple_value=val * 1. / step)
        summary_writer.add_summary(s, global_step)
        if summary_val is not None:
            summary_writer.add_summary(summary_val, global_step)

class Trainer(object):
    def __init__(self, train_model, valid_model, train_data, init_lr=1e-3):

        self._t_model = train_model
        self._v_model = valid_model
        self._train_data = train_data
        self._init_lr = init_lr

        self._train_op = train_model.get_train_op()
        self._train_loss_op = train_model.get_loss()
        self._train_accuracy_op = train_model.get_accuracy()

        self._valid_loss_op = valid_model.get_loss()
        self._valid_accuracy_op = valid_model.get_accuracy()
        # self._train_summary_op = train_model.get_train_summary()
        # self._valid_summary_op = train_model.get_valid_summary()

        self.global_step = 0
        self.epoch_id = 0

    def train_epoch(self, sess, keep_prob=1., summary_writer=None):
        if self.epoch_id < 35:
            self._lr = self._init_lr
        elif self.epoch_id < 50:
            self._lr = self._init_lr / 10.
        else:
            self._lr = self._init_lr / 100.
        # self._t_model.set_is_training(True)
        display_name_list = ['loss', 'accuracy']
        cur_summary = None

        cur_epoch = self._train_data.epochs_completed

        step = 0
        loss_sum = 0
        acc_sum = 0
        self.epoch_id += 1
        while cur_epoch == self._train_data.epochs_completed:
            self.global_step += 1
            step += 1

            batch_data = self._train_data.next_batch_dict()
            im = batch_data['image']
            label = batch_data['label']
            _, loss, acc = sess.run(
                [self._train_op, self._train_loss_op, self._train_accuracy_op], 
                feed_dict={self._t_model.image: im,
                           self._t_model.label: label,
                           self._t_model.lr: self._lr,
                           self._t_model.keep_prob: keep_prob})

            loss_sum += loss
            acc_sum += acc

            if step % 100 == 0:
                display(self.global_step,
                    step,
                    [loss_sum, acc_sum],
                    display_name_list,
                    'train',
                    summary_val=cur_summary,
                    summary_writer=summary_writer)

        print('==== epoch: {}, lr:{} ===='.format(cur_epoch, self._lr))
        display(self.global_step,
                step,
                [loss_sum, acc_sum],
                display_name_list,
                'train',
                summary_val=cur_summary,
                summary_writer=summary_writer)

    def valid_epoch(self, sess, dataflow, summary_writer=None):
        display_name_list = ['loss', 'accuracy']
        cur_summary = None
        dataflow.reset_epoch()

        step = 0
        loss_sum = 0
        acc_sum = 0
        while dataflow.epochs_completed < 1:
            step += 1

            batch_data = dataflow.next_batch_dict()
            im = batch_data['image']
            label = batch_data['label']
            loss, acc = sess.run(
                [self._valid_loss_op, self._valid_accuracy_op], 
                feed_dict={self._v_model.image: im,
                           self._v_model.label: label})

            loss_sum += loss
            acc_sum += acc

        print('[Valid]: ', end='')
        display(self.global_step,
                step,
                [loss_sum, acc_sum],
                display_name_list,
                'valid',
                summary_val=cur_summary,
                summary_writer=summary_writer)
