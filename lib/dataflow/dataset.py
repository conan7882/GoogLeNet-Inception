#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dataset.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np
import copy

from tensorcv.dataflow.image import ImageLabelFromFile, ImageFromFile
from tensorcv.dataflow.common import dense_to_one_hot
from tensorcv.dataflow.base import DataFlow
from tensorcv.utils.utils import assert_type


def separate_data(dataflow, separate_ratio=0.5, class_base=False, shuffle=False):
    assert_type(dataflow, DataFlow)
    assert separate_ratio > 0 and separate_ratio < 1,\
        'separate_ratio must be within (0, 1)!'
    if shuffle:
        dataflow.suffle_data()
    o_data_list = dataflow.get_data_list()
    o_label_list = dataflow.get_label_list()

    if class_base:
        print('*** Not Implemented ! ***')
    else:
        n_data = len(o_data_list)
        n_part_1 = int(np.ceil(n_data * separate_ratio))
        dataflow_1 = copy.copy(dataflow)
        dataflow_2 = copy.copy(dataflow)
        dataflow_1.set_data_list(o_data_list[:n_part_1])
        dataflow_1.set_label_list(o_label_list[:n_part_1])
        dataflow_2.set_data_list(o_data_list[n_part_1:])
        dataflow_2.set_label_list(o_label_list[n_part_1:])

    return dataflow_1, dataflow_2


class ImageLabelFromCSVFile(ImageLabelFromFile):
    def __init__(self, ext_name, data_dir='', 
                 label_file_name='', start_line=0,
                 num_channel=None, one_hot=False,
                 label_dict={}, num_class=None,
                 shuffle=True, normalize=None,
                 resize=None):
        self._start_line = start_line
        super(ImageLabelFromCSVFile, self).__init__(
            ext_name, data_dir=data_dir, 
            label_file_name=label_file_name,
            num_channel=num_channel, one_hot=one_hot,
            label_dict=label_dict, num_class=num_class,
            shuffle=shuffle, normalize=normalize,
            resize=resize)

    def _load_file_list(self, ext_name):
        label_file = open(
            os.path.join(self.data_dir, self._label_file_name),'r')
        lines = label_file.read().split('\n')[self._start_line:]
        
        self._im_list = np.array([self.data_dir + line.split(',')[0] + ext_name
                         for line in lines 
                         if len(line.split(',')) == 2])
        label_list = np.array([line.split(',')[1]
                         for line in lines 
                         if len(line.split(',')) == 2])
        label_file.close()

        if self.label_dict is None or not bool(self.label_dict):
            self.label_dict = {}
            label_cnt = 0
            for cur_label in label_list:
                if not cur_label in self.label_dict:
                    self.label_dict[cur_label] = label_cnt
                    label_cnt += 1
        if self._num_class is None:
            self._num_class = len(self.label_dict)
        
        self._label_list = np.array([self.label_dict[cur_label] 
                                     for cur_label in label_list])

        if self._one_hot:
            self._label_list = dense_to_one_hot(self._label_list, self._num_class)

        # if self._shuffle:
        #     self._suffle_file_list()

    def get_data_list(self):
        return self._im_list

    def set_data_list(self, new_list):
        self._im_list = new_list

    def get_label_list(self):
        return self._label_list

    def set_label_list(self, new_list):
        self._label_list = new_list

    def suffle_data(self):
        self._suffle_file_list()


class new_ImageFromFile(ImageFromFile):
    def next_batch(self):
        assert self._batch_size <= self.size(), \
        "batch_size cannot be larger than data size"

        if self._data_id + self._batch_size > self.size():
            start = self._data_id
            end = self.size()
            print(end)
            self._epochs_completed += 1
            self._data_id = 0
            if self._shuffle:
                self._suffle_file_list()
        else:

            start = self._data_id
            self._data_id += self._batch_size
            end = self._data_id
        # batch_file_range = range(start, end)

        return self._load_data(start, end)


if __name__ == '__main__':
    data_dir = '/Users/gq/workspace/Dataset/kaggle/dog_bleed/train/'
    d = ImageLabelFromCSVFile('.jpg', data_dir=data_dir, start_line=1,
                              label_file_name='../labels.csv',
                              num_channel=3)

    # print(d.label_dict)
    d_1, d_2 = separate_data(d, separate_ratio=0.7, class_base=False)
    print(len(set(d_1._label_list)), len(set(d_2._label_list)), d.size())

