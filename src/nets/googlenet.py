#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: googlenet.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf

from src.nets.base import BaseModel
import src.models.layers as L
import src.models.inception_module as module
import src.utils.viz as viz
import src.utils.multigpu as multigpu


INIT_W = tf.keras.initializers.he_normal()

class GoogLeNet(BaseModel):
    """ base model of GoogleNet for image classification """

    def __init__(self, n_channel, n_class, pre_trained_path=None,
                 bn=False, wd=0, conv_trainable=True, fc_trainable=True,
                 sub_imagenet_mean=True, var_device=None):
        self._n_channel = n_channel
        self.n_class = n_class
        self._bn = bn
        self._wd = wd
        self._conv_trainable = conv_trainable
        self._fc_trainable = fc_trainable
        self._sub_imagenet_mean = sub_imagenet_mean

        self.var_device = var_device

        self._pretrained_dict = None
        if pre_trained_path:
            self._pretrained_dict = np.load(
                pre_trained_path, encoding='latin1').item()

        self.layers = {}

    def _create_train_input(self, device_id):
        self.layers['image_{}'.format(device_id)] = tf.placeholder(
            tf.float32, [None, None, None, self._n_channel], name='image')
        self.layers['label_{}'.format(device_id)] = tf.placeholder(tf.int64, [None], 'label')
        self.layers['keep_prob_{}'.format(device_id)] = tf.placeholder(tf.float32, name='keep_prob')
        
    def _create_test_input(self):
        self.image = tf.placeholder(
            tf.float32, [None, None, None, self._n_channel], name='image')
        self.label = tf.placeholder(tf.int64, [None], 'label')
        self.keep_prob = 1.

    def _create_train_inference(self, device_id):
        
        if self._sub_imagenet_mean:
            net_input = module.sub_rgb2bgr_mean(self.layers['image_{}'.format(device_id)])
        else:
            net_input = self.layers['image_{}'.format(device_id)]

        with tf.variable_scope('conv_layers', reuse=tf.AUTO_REUSE):
            self.layers['conv_out_{}'.format(device_id)] = self._conv_layers(net_input)

        with tf.variable_scope('inception_layers', reuse=tf.AUTO_REUSE):
            self.layers['inception_out_{}'.format(device_id)] = self._inception_layers(
                self.layers['conv_out_{}'.format(device_id)])

        with tf.variable_scope('fc_layers', reuse=tf.AUTO_REUSE):   
            self.layers['logits_{}'.format(device_id)] = self._fc_layers(
                self.layers['inception_out_{}'.format(device_id)],
                keep_prob=self.layers['keep_prob_{}'.format(device_id)])

        with tf.variable_scope('auxiliary_classifier_0', reuse=tf.AUTO_REUSE):
            self.layers['auxiliary_logits_0_{}'.format(device_id)] = self._auxiliary_classifier(
                self.layers['inception_4a'],
                keep_prob=self.layers['keep_prob_{}'.format(device_id)])

        with tf.variable_scope('auxiliary_classifier_1', reuse=tf.AUTO_REUSE):
            self.layers['auxiliary_logits_1_{}'.format(device_id)] = self._auxiliary_classifier(
                self.layers['inception_4d'],
                keep_prob=self.layers['keep_prob_{}'.format(device_id)])


    def create_train_model(self, gpu_list=None):
        self.set_is_training(is_training=True)

        if gpu_list is None:
            gpu_list = [0]
        elif not isinstance(gpu_list, list):
            gpu_list = gpu_list


        with tf.device('/cpu:0') as scope:
            # self.opt = self._get_optimizer()
            tower_grads = []
            tower_loss = []
            tower_accuracy = []

            self.lr = tf.placeholder(tf.float32, name='lr')
            opt = self.get_optimizer()

            for gpu_id in gpu_list:
                with tf.device('/gpu:{}'.format(gpu_id)) as scope:
                    self._create_train_input(device_id=gpu_id)
                    self._create_train_inference(device_id=gpu_id)
                    tower_accuracy.append(self.get_accuracy(device_id=gpu_id))
                    with tf.name_scope('train_{}'.format(gpu_id)):
                        loss = self._get_loss(device_id=gpu_id)
                        tower_loss.append(loss)
                        var_list = tf.trainable_variables()
                        grads = tf.gradients(loss, var_list)
                        tower_grads.append(zip(grads, var_list))

            grads = multigpu.average_gradients(tower_grads)     
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = opt.apply_gradients(grads)
                
        # self.train_op = self.get_train_op()
            self.loss_op = tf.reduce_mean(tf.stack(tower_loss, axis=0), axis=0)
            self.accuracy_op = tf.reduce_mean(tf.stack(tower_accuracy, axis=0), axis=0)

        self.global_step = 0
        self.epoch_id = 0

    def create_test_model(self):
        self.set_is_training(is_training=False)
        self._create_test_input()
        if self._sub_imagenet_mean:
            net_input = module.sub_rgb2bgr_mean(self.image)
        else:
            net_input = self.image

        with tf.variable_scope('conv_layers', reuse=tf.AUTO_REUSE):
            self.layers['conv_out'] = self._conv_layers(net_input)
        with tf.variable_scope('inception_layers', reuse=tf.AUTO_REUSE):
            self.layers['inception_out'] = self._inception_layers(self.layers['conv_out'])
        with tf.variable_scope('fc_layers', reuse=tf.AUTO_REUSE):   
            self.layers['logits'] = self._fc_layers(self.layers['inception_out'])
            self.layers['top_5'] = tf.nn.top_k(
                tf.nn.softmax(self.layers['logits']), k=5, sorted=True)

        self.loss_op = self.get_loss()
        self.accuracy_op = self.get_accuracy()

        self.global_step = 0
        self.epoch_id = 0

    def _conv_layers(self, inputs):
        conv_out = module.inception_conv_layers(
            layer_dict=self.layers, inputs=inputs,
            pretrained_dict=self._pretrained_dict,
            bn=self._bn, wd=self._wd, init_w=INIT_W,
            is_training=self.is_training, trainable=self._conv_trainable, var_device=self.var_device)
        return conv_out

    def _inception_layers(self, inputs):
        inception_out = module.inception_layers(
            layer_dict=self.layers, inputs=inputs,
            pretrained_dict=self._pretrained_dict,
            bn=self._bn, wd=self._wd, init_w=INIT_W,
            is_training=self.is_training, trainable=self._conv_trainable, var_device=self.var_device)
        return inception_out

    def _fc_layers(self, inputs, keep_prob):
        fc_out = module.inception_fc(
            layer_dict=self.layers, n_class=self.n_class, keep_prob=keep_prob,
            inputs=inputs, pretrained_dict=self._pretrained_dict,
            bn=self._bn, init_w=INIT_W, trainable=self._fc_trainable,
            is_training=self.is_training, wd=self._wd, var_device=self.var_device)
        return fc_out

    def _auxiliary_classifier(self, inputs, keep_prob):
        logits = module.auxiliary_classifier(
            layer_dict=self.layers, n_class=self.n_class, keep_prob=keep_prob,
            inputs=inputs, pretrained_dict=None, is_training=self.is_training,
            bn=self._bn, init_w=INIT_W, trainable=self._fc_trainable, wd=self._wd, var_device=self.var_device)
        return logits

    def _get_loss(self, device_id):
        with tf.name_scope('loss'):
            labels = self.layers['label_{}'.format(device_id)]
            logits = self.layers['logits_{}'.format(device_id)]
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits,
                name='cross_entropy')
            cross_entropy = tf.reduce_mean(cross_entropy)
        if self.is_training:
            auxilarity_loss = self._get_auxiliary_loss(0, device_id) + self._get_auxiliary_loss(1, device_id)
            return cross_entropy + 0.3 * auxilarity_loss
        else:
            return cross_entropy

    def _get_auxiliary_loss(self, loss_id, device_id):
        with tf.name_scope('auxilarity_loss_{}'.format(loss_id)):
            labels = self.layers['label_{}'.format(device_id)]
            logits = self.layers['auxiliary_logits_{}_{}'.format(loss_id, device_id)]
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits,
                name='cross_entropy')
        return tf.reduce_mean(cross_entropy)

    def _get_optimizer(self):

        return tf.train.AdamOptimizer(self.lr)

    def get_accuracy(self, device_id):
        with tf.name_scope('accuracy'):
            prediction = tf.argmax(self.layers['logits_{}'.format(device_id)], axis=1)
            correct_prediction = tf.equal(prediction, self.layers['label_{}'.format(device_id)])
            return tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), 
                name = 'result')

    def train_epoch(self, sess, init_lr, train_data, keep_prob, summary_writer=None):
        if self.epoch_id < 35:
            lr = init_lr
        elif self.epoch_id < 50:
            lr = init_lr / 10.
        else:
            lr = init_lr / 100.

        display_name_list = ['loss', 'accuracy']
        cur_summary = None

        cur_epoch = train_data.epochs_completed

        step = 0
        loss_sum = 0
        acc_sum = 0
        self.epoch_id += 1
        while cur_epoch == train_data.epochs_completed:
            self.global_step += 1
            step += 1

            feed_dict = {self.lr: lr}
            for device_id in range(2):
                batch_data = train_data.next_batch_dict()
                im = batch_data['image']
                label = batch_data['label']

                feed_dict[self.layers['image_{}'.format(device_id)]] = im
                feed_dict[self.layers['label_{}'.format(device_id)]] = label
                feed_dict[self.layers['keep_prob_{}'.format(device_id)]] = keep_prob

            _, loss, acc = sess.run(
                [self.train_op, self.loss_op, self.accuracy_op], 
                feed_dict=feed_dict)

            loss_sum += loss
            acc_sum += acc

            if step % 100 == 0:
                viz.display(self.global_step,
                    step,
                    [loss_sum, acc_sum],
                    display_name_list,
                    'train',
                    summary_val=cur_summary,
                    summary_writer=summary_writer)

        print('==== epoch: {}, lr:{} ===='.format(cur_epoch, lr))
        viz.display(
            self.global_step,
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
                [self.loss_op, self.accuracy_op], 
                feed_dict={self.image: im,
                           self.label: label})

            loss_sum += loss
            acc_sum += acc

        print('[Valid]: ', end='')
        viz.display(
            self.global_step,
            step,
            [loss_sum, acc_sum],
            display_name_list,
            'valid',
            summary_val=cur_summary,
            summary_writer=summary_writer)


class GoogLeNet_cifar(GoogLeNet):
    def _fc_layers(self, inputs, keep_prob):
        fc_out = module.inception_fc(
            layer_dict=self.layers, n_class=self.n_class, keep_prob=keep_prob,
            inputs=inputs, pretrained_dict=None,
            bn=self._bn, init_w=INIT_W, trainable=self._fc_trainable,
            is_training=self.is_training, wd=self._wd, var_device=self.var_device)
        return fc_out

    def _conv_layers(self, inputs):
        conv_out = module.inception_conv_layers_cifar(
            layer_dict=self.layers, inputs=inputs,
            pretrained_dict=None,
            bn=self._bn, wd=self._wd, init_w=INIT_W,
            is_training=self.is_training, trainable=self._conv_trainable,
            conv_stride=1, var_device=self.var_device)
        return conv_out

