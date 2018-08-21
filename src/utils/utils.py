#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: utils.py
# Author: Qian Ge <geqian1001@gmail.com>


def make_list(inputs):
    if not isinstance(inputs, list):
        return [inputs]
    else:
        return inputs

def assert_len(check_list):
    for ele in check_list[1:]:
        assert len(check_list[0]) == len(ele)