#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: pre_trained.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import numpy as np
import scipy.misc

from tensorcv.dataflow.image import ImageFromFile

import setup_env
from models.googlenet import GoogleNet
from utils.preprocess import resize_image_with_smallest_side, center_crop_image
from utils.classes import get_word_list


def display_data(dataflow, data_name):
    try:
        print('[{}] num of samples {}, num of classes {}'.\
            format(data_name, dataflow.size(), len(dataflow.label_dict)))
    except AttributeError:
        print('[{}] num of samples {}'.\
            format(data_name, dataflow.size()))
    print(dataflow._im_list)


if __name__ == '__main__':
    SAVE_DIR = 'E:/tmp/cnn/'
    PARA_DIR = 'E:/GITHUB/workspace/CNN/pretrained/googlenet.npy'
    DATA_DIR = '../data/'

    model = GoogleNet(is_load=True, pre_train_path=PARA_DIR)

    image = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    test_data = ImageFromFile('.jpeg', data_dir=DATA_DIR, num_channel=3)
    display_data(test_data, 'test_data')

    word_dict = get_word_list('../data/imageNetLabel.txt')

    model.create_model([image, 1])
    test_op = tf.nn.top_k(tf.nn.softmax(model.layer['output']),
                             k=5, sorted=True)
    input_op = model.layer['input']

    writer = tf.summary.FileWriter(SAVE_DIR)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        for k in range(0, 50):
            if test_data.epochs_completed < 1:
                batch_data = test_data.next_batch()
                im = batch_data[0]
                im = resize_image_with_smallest_side(im, 224)
                im = center_crop_image(im, 224, 224)
                scipy.misc.imsave('{}test_{}.png'.format(SAVE_DIR, k), np.squeeze(im))
                result = sess.run(test_op, feed_dict={image: im})
                for val, ind in zip(result.values, result.indices):
                    print(val)
                    print(ind)
                    print(word_dict[ind[0]])
