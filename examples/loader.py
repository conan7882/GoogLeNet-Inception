#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: loader.py
# Author: Qian Ge <geqian1001@gmail.com>

import sys
import numpy as np
import tensorflow as tf
import skimage.transform

sys.path.append('../')
from src.dataflow.images import Image
from src.dataflow.cifar import CIFAR 

def load_label_dict(dataset='imagenet'):
    """ 
        Function to read the ImageNet label file.
        Used for testing the pre-trained model.

        dataset (str): name of data set. 'imagenet', 'cifar'
    """
    label_dict = {}
    if dataset == 'cifar':
        file_path = '../data/cifarLabel.txt'
    else:
        file_path = '../data/imageNetLabel.txt'
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            if dataset == 'cifar':
                names = line.rstrip()
            else:
                # point to the class label string
                names = line.rstrip()[10:]
            label_dict[idx] = names
    return label_dict

def read_image(im_name, n_channel, data_dir='', batch_size=1, rescale=True):
    """ function for create a Dataflow for reading images from a folder
        This function returns a Dataflow object for images with file 
        name containing 'im_name' in directory 'data_dir'.

        Args:
            im_name (str): part of image names (i.e. 'jpg' or 'im_').
            n_channel (int): number of channels (3 for color images and 1 for grayscale images)
            data_dir (str): directory of images
            batch_size (int): number of images read from Dataflow for each batch
            rescale (bool): whether rescale image to 224 or not

        Returns:
            Image (object): batch images can be access by Image.next_batch_dict()['image']
    """

    def rescale_im(im):
        """ Pre-process for images 
            images are rescaled so that the shorter side = 224
        """
        im = np.array(im)
        h, w = im.shape[0], im.shape[1]
        if h >= w:
            new_w = 224
            im = skimage.transform.resize(im, (int(h * new_w / w), 224),
                                          preserve_range=True)
        else:
            new_h = 224
            im = skimage.transform.resize(im, (224, int(w * new_h / h)),
                                          preserve_range=True)
        return im.astype('uint8')

    if rescale:
        pf_fnc = rescale_im
    else:
        pf_fnc = None

    image_data = Image(
        im_name=im_name,
        data_dir=data_dir,
        n_channel=n_channel,
        shuffle=False,
        batch_dict_name=['image'],
        pf_list=pf_fnc)
    image_data.setup(epoch_val=0, batch_size=batch_size)

    return image_data

def load_cifar(cifar_path, batch_size=64, subtract_mean=True):
    """ function for create Dataflow objects for CIFAR-10

        Args:
            cifar_path (str): directory of CIFAR-10 data
            batch_size (int): number of images read from Dataflow for each batch
            substract_mean (bool): whether subtract each channel by average of training set

        Returns:
            CIFAR (object) of training and testing set.
            Batch images and label can be access by
            CIFAR.next_batch_dict()['image'] and 
            CIFAR.next_batch_dict()['label']
    """

    train_data = CIFAR(
        data_dir=cifar_path,
        shuffle=True,
        batch_dict_name=['image', 'label'],
        data_type='train',
        channel_mean=None,
        subtract_mean=subtract_mean,
        augment=True,
        # pf=preprocess,
        )
    train_data.setup(epoch_val=0, batch_size=batch_size)

    valid_data = CIFAR(
        data_dir=cifar_path,
        shuffle=False,
        batch_dict_name=['image', 'label'],
        data_type='valid',
        channel_mean=train_data.channel_mean,
        subtract_mean=subtract_mean,
        augment=False,
        # pf=pf_test,
        )
    valid_data.setup(epoch_val=0, batch_size=batch_size)

    return train_data, valid_data

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image_data, test_data = load_cifar(
        'E:/Dataset/cifar/', batch_size=100, subtract_mean=True)
    batch_data = image_data.next_batch_dict()
    print(batch_data['image'].shape)
    plt.figure()
    plt.imshow(np.squeeze(batch_data['image'][0]))
    print(type(batch_data['image'][0]))

    plt.figure()
    plt.imshow(np.squeeze(batch_data['image'][1]))


    batch_data = image_data.next_batch_dict()
    plt.figure()
    plt.imshow(np.squeeze(batch_data['image'][0]))
    print(type(batch_data['image'][0]))
    plt.figure()
    plt.imshow(np.squeeze(batch_data['image'][1]))

    plt.show()
    