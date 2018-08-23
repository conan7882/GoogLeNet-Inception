#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cifar.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import pickle
import numpy as np
import tensorflow as tf

from src.dataflow.base import DataFlow


class CIFAR(DataFlow):
    def __init__(self,
                 data_dir='',
                 shuffle=True,
                 batch_dict_name=None,
                 data_type='train',
                 channel_mean=None,
                 subtract_mean=True,
                 pf=None,
                 augment=False):
        self._mean = channel_mean
        self._subtract = subtract_mean
        self._pf = pf

        self._augment = augment
        if augment:
            self._augment_flow = tf.keras.preprocessing.image.ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                zca_epsilon=1e-06,
                rotation_range=20.0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                brightness_range=None,
                shear_range=0.0,
                zoom_range=0.0,
                channel_shift_range=0.0,
                fill_mode='nearest',
                cval=0.0,
                horizontal_flip=True,
                vertical_flip=False,
                rescale=None,
                preprocessing_function=None,
                data_format=None,
                validation_split=0.0)
        # self.num_channels = 3
        # self.im_size = [32, 32]

        # assert os.path.isdir(data_dir)
        # self.data_dir = data_dir

        assert batch_dict_name is not None
        if not isinstance(batch_dict_name, list):
            batch_dict_name = [batch_dict_name]
        self._batch_dict_name = batch_dict_name

        if data_type == 'train':
            self._file_list = [os.path.join(data_dir, 'data_batch_{}'.format(i)) for i in range(1, 6)]
        else:
            self._file_list = [os.path.join(data_dir, 'test_batch')]

        self.shuffle = shuffle

        self.setup(epoch_val=0, batch_size=1)
        # if not isinstance(batch_file_list, list):
        #     batch_file_list = [batch_file_list]
        # self._file_list = [os.path.join(data_dir, 'data_batch_' + str(batch_id)) for batch_id in batch_file_list]

        # self._load_files()
        # self._num_image = self.size()

        self._image_id = 0
        self._batch_file_id = -1
        self._image = []
        self._next_batch_file()

        print('Data Loaded! Size of data: {}'.format(self.size()))

    def _next_batch_file(self):
        if self._batch_file_id >= len(self._file_list) - 1:
            self._batch_file_id = 0
            self._epochs_completed += 1
        else:
            self._batch_file_id += 1
        data_dict = unpickle(self._file_list[self._batch_file_id])
        self._image = np.array(data_dict['image'])
        if self._pf:
            self._image = [self._pf(im) for im in self._image]
            self._image = np.array(self._image )
        self._label = np.array(data_dict['label'])

        if self.shuffle:
            self._suffle_files()

    def _suffle_files(self):
        idxs = np.arange(len(self._image))

        self.rng.shuffle(idxs)
        self._image = self._image[idxs]
        self._label = self._label[idxs]

    @property
    def batch_step(self):
        return int(self.size() * 1.0 / self._batch_size)

    @property
    def channel_mean(self):
        if self._mean == None:
            self._mean = self._comp_channel_mean()
        return self._mean

    def subtract_channel_mean(self, im_list):
        """
        Args:
            im_list: [batch, h, w, c]
        """
        mean = self.channel_mean
        for c_id in range(0, im_list.shape[-1]):
            im_list[:, :, :, c_id] = im_list[:, :, :, c_id] - mean[c_id]
        return im_list

    def _comp_channel_mean(self):
        im_list = []
        for k in range(len(self._file_list)):
            cur_im = unpickle(self._file_list[k])['image']
            im_list.extend(cur_im)
        im_list = np.array(im_list)

        mean_list = []
        for c_id in range(0, im_list.shape[-1]):
            mean_list.append(np.mean(im_list[:,:,:,c_id]))
        return mean_list

    def size(self):
        try:
            return self.data_size
        except AttributeError:
            data_size = 0
            for k in range(len(self._file_list)):
                tmp_image = unpickle(self._file_list[k])['image']
                data_size += len(tmp_image)
            self.data_size = data_size
            return self.data_size
        
    def next_batch(self):
        assert self._batch_size <= self.size(), \
          "batch_size {} cannot be larger than data size {}".\
           format(self._batch_size, self.size())

        start = self._image_id
        self._image_id += self._batch_size
        end = self._image_id
        batch_image = np.array(self._image[start:end])
        batch_label = np.array(self._label[start:end])

        if self._image_id + self._batch_size > len(self._image):
            self._next_batch_file()
            self._image_id = 0
            if self.shuffle:
                self._suffle_files()
        if self._augment:
            self._augment_flow.fit(batch_image)
            batch_image, batch_label = self._augment_flow.flow(batch_image, batch_label, batch_size=self._batch_size)[0]
        if self._subtract:
            batch_image = self.subtract_channel_mean(batch_image)
        # batch_image = batch_image.astype('float32')
        # batch_image = batch_image / 255. * 2. - 1.
        return batch_image.astype('float32'), batch_label

    def next_batch_dict(self):
        re_dict = {}
        batch_data = self.next_batch()
        for key, data in zip(self._batch_dict_name, batch_data):
            re_dict[key] = data
        return re_dict


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    image = dict[b'data']
    labels = dict[b'labels']

    r = image[:,:32*32].reshape(-1,32,32)
    g = image[:,32*32: 2*32*32].reshape(-1,32,32)
    b = image[:,2*32*32:].reshape(-1,32,32)

    image = np.stack((r,g,b),axis=-1)

    return {'image': image.astype(float), 'label': labels}

