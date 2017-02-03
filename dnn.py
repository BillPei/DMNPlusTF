from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

l2_reg = tf.contrib.layers.l2_regularizer(1.0)

def random_uniform_init(range_num, seed=None, dtype=tf.float32):
    def _initializer(shape, **kwargs):
        return tf.random_uniform(shape=shape, minval=-range_num, maxval=range_num, dtype=dtype, seed=seed)
    return _initializer

def matmul_3d_2d(X_3d, W_2d, bias_to_add=None):
    #bias can be 1d or 0d
    out = tf.einsum('ijk,kl->ijl', X_3d, W_2d)
    if bias_to_add is not None:
        out += bias_to_add
    return out