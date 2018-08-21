#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np 
import src.utils.utils as utils
from src.utils.dataflow import get_rng, get_file_list


class DataFlow(object):
    def __init__(self,
                 data_name_list,
                 data_dir='',
                 shuffle=True,
                 batch_dict_name=None,
                 load_fnc_list=None,
                 ):
        data_name_list = utils.make_list(data_name_list)
        load_fnc_list = utils.make_list(load_fnc_list)
        data_dir = utils.make_list(data_dir)

        if len(data_dir) == 1:
            data_dir_list = [data_dir[0] for i in range(len(load_fnc_list))]
            data_dir = data_dir_list

        dataflow_list = []
        self._load_fnc_list = []
        for data_name, load_fnc in zip(data_name_list, load_fnc_list):
            if data_name is not None and load_fnc is not None:
                dataflow_list.append(data_name)
                self._load_fnc_list.append(load_fnc)
            else:
                break
        self._n_dataflow = len(dataflow_list)

        # pf_list = utils.make_list(pf_list)

        # utils.assert_len([data_name_list, load_fnc_list])
        # self._n_dataflow = len(data_name_list)
        # self._load_fnc_list = load_fnc_list

        # self._data_dir = data_dir
        self._shuffle = shuffle
        self._batch_dict_name = batch_dict_name

        self._data_id = 0
        self.setup(epoch_val=0, batch_size=1)
        self._load_file_list(dataflow_list, data_dir)
        self._cur_file_name = [[] for i in range(len(self._file_name_list))]

    def setup(self, epoch_val, batch_size, **kwargs):
        self._epochs_completed  = epoch_val
        self._batch_size = batch_size
        self.rng = get_rng(self)
        self._setup()

    def reset_epoch(self):
        self._epochs_completed = 0

    def _setup(self):
        pass

    def size(self):
        return len(self._file_name_list[0])

    def _load_file_list(self, data_name_list, data_dir_list):
        self._file_name_list = []
        for data_name, data_dir in zip(data_name_list, data_dir_list):
            self._file_name_list.append(get_file_list(data_dir, data_name))
        if self._shuffle:
            self._suffle_file_list()


    def _suffle_file_list(self):
        idxs = np.arange(self.size())
        self.rng.shuffle(idxs)
        for idx, file_list in enumerate(self._file_name_list):
            self._file_name_list[idx] = file_list[idxs]

    def next_batch(self):
        assert self._batch_size <= self.size(), \
        "batch_size cannot be larger than data size"

        if self._data_id + self._batch_size > self.size():
            start = self._data_id
            end = self.size()
        else:
            start = self._data_id
            self._data_id += self._batch_size
            end = self._data_id
        batch_data = self._load_data(start, end)

        for flow_id in range(len(self._file_name_list)):
            self._cur_file_name[flow_id] = self._file_name_list[flow_id][start: end]

        if end == self.size():
            self._epochs_completed += 1
            self._data_id = 0
            if self._shuffle:
                self._suffle_file_list()
        return batch_data

    def _load_data(self, start, end):
        data_list = [[] for i in range(0, self._n_dataflow)]
        for k in range(start, end):
            for read_idx, read_fnc in enumerate(self._load_fnc_list):
                data = read_fnc(self._file_name_list[read_idx][k])
                data_list[read_idx].append(data)

        for idx, data in enumerate(data_list):
            data_list[idx] = np.array(data)

        return data_list

    def next_batch_dict(self):
        batch_data = self.next_batch()
        return {key: data for key, data in zip(self._batch_dict_name, batch_data)} 

    def get_batch_file_name(self):
        return self._cur_file_name

    @property
    def epochs_completed(self):
        return self._epochs_completed
