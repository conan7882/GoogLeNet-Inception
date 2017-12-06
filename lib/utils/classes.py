#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: classes.py
# Author: Qian Ge <geqian1001@gmail.com>

import os


def get_word_list(file_path):
    word_dict = {}
    word_file = open(os.path.join(file_path), 'r')
    lines = word_file.read().split('\n')
    for i, line in enumerate(lines):
        label, word = line.split(' ', 1)
        word_dict[i] = word
    return word_dict
