#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: fine_tuning.py
# Author: Qian Ge <geqian1001@gmail.com>

from collections import namedtuple
import argparse

import tensorflow as tf

from tensorcv.train.config import TrainConfig
from tensorcv.predicts.config import PridectConfig
from tensorcv.predicts.predictions import *
from tensorcv.predicts.simple import SimpleFeedPredictor
from tensorcv.train.simple import SimpleFeedTrainer
from tensorcv.callbacks import *
from tensorcv.dataflow.image import ImageFromFile

import sys
sys.path.append('../lib/')
from nets.googlenet_finetune import GoogLeNet_Finetuning
from dataflow.dataset import ImageLabelFromCSVFile, new_ImageFromFile, separate_data

DATA_DIR = '/home/qge2/workspace/data/dataset/dog_bleed/train/'
TEST_DIR = '/home/qge2/workspace/data/dataset/dog_bleed/test/'
PARA_PATH = '/home/qge2/workspace/data/pretrain/inception/googlenet.npy'
SAVE_DIR = '/home/qge2/workspace/data/tmp2/'
configpath = namedtuple('CONFIG_PATH', ['summary_dir', 'checkpoint_dir', 'model_dir', 'result_dir'])
config_path = configpath(summary_dir=SAVE_DIR, checkpoint_dir=SAVE_DIR, model_dir=SAVE_DIR, result_dir=SAVE_DIR)


def config_train(FLAGS):
    dataset = ImageLabelFromCSVFile('.jpg', data_dir=DATA_DIR, start_line=1,
                              label_file_name='../labels.csv',
                              num_channel=3, resize=224)
    train_data, val_data = separate_data(dataset, separate_ratio=0.7,
                                         class_base=False, shuffle=False)

    n_class = len(train_data.label_dict)

    net = GoogLeNet_Finetuning(num_class=n_class,
                               num_channels=3,
                               learning_rate=FLAGS.lr,
                               is_load=True,
                               pre_train_path=PARA_PATH,
                               im_height=224, im_width=224,
                               drop_out=0.3)

    inference_list_validation = InferScalars(['accuracy/result', 'loss/result', 'loss/cross_entropy'],
                                             ['test_accuracy', 'test_loss', 'test_entropy'])

    training_callbacks = [
        ModelSaver(periodic=300),
        TrainSummary(key='train', periodic=10),
        FeedInferenceBatch(val_data, batch_count=20, periodic=10,
                           inferencers=inference_list_validation),
        CheckScalar(['accuracy/result', 'loss/result'], periodic=10)]

    return TrainConfig(
        dataflow=train_data, model=net,
        callbacks=training_callbacks,
        batch_size=32, max_epoch=25,
        monitors=TFSummaryWriter(),
        summary_periodic=10,
        is_load=False,
        model_name='gap_1024_1024/model-5700',
        default_dirs=config_path)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--predict', action='store_true',
                        help='Run prediction')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')

    parser.add_argument('--lr', default=1e-6, type=float,
                        help='learning rate of fine tuning')

    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.train:
        config = config_train(FLAGS)
        SimpleFeedTrainer(config).train()
    if FLAGS.predict:
        config = config_predict()
        SimpleFeedPredictor(config).run_predict()
