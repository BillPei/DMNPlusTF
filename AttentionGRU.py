from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.nn import tanh
from dnn import l2_reg

class AttentionGRU(object):
    def __init__(self, hidden_size, initializer=tf.contrib.layers.xavier_initializer(), scope=None):
        self._hidden_size = hidden_size
        self.initializer = initializer

        with tf.variable_scope(scope or 'AttentionGRU'):
            with tf.variable_scope("Gates"):  # Reset gate
                # todo:  start with bias of 1.0 to not reset??
                #r: batch_size, hidden_size
                self.w_r = tf.get_variable("W", shape=[hidden_size, hidden_size], dtype=tf.float32, initializer=self.initializer,
                                    regularizer=l2_reg, trainable=True)
                self.u_r = tf.get_variable("U", shape=[hidden_size, hidden_size], dtype=tf.float32, initializer=self.initializer,
                                    regularizer=l2_reg, trainable=True)
                self.b_r = tf.get_variable("bias_", shape=hidden_size, dtype=tf.float32, initializer=tf.constant_initializer(0.0),
                                    trainable=True)
            with tf.variable_scope("Candidate"):
                self.w_c = tf.get_variable("W", shape=[hidden_size, hidden_size], dtype=tf.float32,
                                      initializer=self.initializer,
                                      regularizer=l2_reg, trainable=True)
                self.u_c = tf.get_variable("U", shape=[hidden_size, hidden_size], dtype=tf.float32,
                                      initializer=self.initializer,
                                      regularizer=l2_reg, trainable=True)
                self.b_c = tf.get_variable("bias_", shape=hidden_size, dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.0),
                                      trainable=True)


    def __call__(self, inputs, state, attention, scope=None):
        #input: batch_size, hidden_size
        #state: batch_size, hidden_size
        #attention: batch_size, 1

        # batch_size, hidden_size
        r = tf.nn.sigmoid(tf.matmul(inputs, self.w_r) + tf.matmul(state, self.u_r) + self.b_r)
        # batch_size, hidden_size
        c = tanh(tf.matmul(inputs, self.w_c) + tf.matmul(r*state, self.u_c) + self.b_c)
        # new_h: batch_size, hidden_size
        new_h = attention * c + (1 - attention) * state

        return new_h
