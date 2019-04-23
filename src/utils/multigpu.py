#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: multigpu.py
# Author: Qian Ge <geqian1001@gmail.com>


import tensorflow as tf


def average_gradients(tower_grads):
        # https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py#L110
        """Calculate the average gradient for each shared variable across all towers.
            Note that this function provides a synchronization point across all towers.
            Args:
                tower_grads: List of lists of (gradient, variable) tuples. The outer list
                    is over individual gradients. The inner list is over the gradient
                    calculation for each tower.
            Returns:
                List of pairs of (gradient, variable) where the gradient has been averaged
                    across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.stack(grads, axis=0)
            grad = tf.squeeze(tf.reduce_mean(grad, axis=0), axis=0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads