#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: images.py
# Author: Qian Ge <geqian1001@gmail.com>

from src.dataflow.base import DataFlow
from src.utils.dataflow import load_image, identity, fill_pf_list


class Image(DataFlow):
    def __init__(self,
                 im_name,
                 data_dir='',
                 n_channel=3,
                 shuffle=True,
                 batch_dict_name=None,
                 pf_list=None):
        pf_list = fill_pf_list(
            pf_list, n_pf=1, fill_with_fnc=identity)

        def read_image(file_name):
            im = load_image(file_name, read_channel=n_channel,  pf=pf_list[0])
            return im

        super(Image, self).__init__(
            data_name_list=[im_name],
            data_dir=data_dir,
            shuffle=shuffle,
            batch_dict_name=batch_dict_name,
            load_fnc_list=[read_image],
            ) 