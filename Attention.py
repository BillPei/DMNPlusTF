from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from dnn import matmul_3d_2d
from dnn import l2_reg


class DMNAttentionGate:
    """
    Possible enhancements
    1. make scope as an input param
    2. get shape from inputs itself
    """
    def __init__(self, hidden_size, num_attention_features, dtype=tf.float32,  initializer=tf.contrib.layers.xavier_initializer()):
        with tf.variable_scope("AttentionGate"):
            self.b_1 = tf.get_variable("bias_1", (hidden_size,), dtype=dtype, initializer=tf.constant_initializer(0.0))
            self.W_1 = tf.get_variable("W_1", (hidden_size*num_attention_features, hidden_size), dtype=dtype, initializer=initializer, regularizer=l2_reg)

            self.W_2 = tf.get_variable("W_2", (hidden_size, 1), dtype=dtype, initializer=initializer, regularizer=l2_reg)
            self.b_2 = tf.get_variable("bias_2", 1, dtype=dtype, initializer=tf.constant_initializer(0.0))

    def get_attention(self, questions, prev_memory, facts, sentences_per_input_in_batch):
        #questions_gru_final_state dim: batch_size, hidden_size
        #facts_from_fusion_layer dim: [max_sentences_per_input, [batch_size, hidden_size]]
        with tf.variable_scope("AttentionGate"):
            # features dim: list of length 4, each [max_sentences_per_input, batch_size, hidden_size]
            features = [facts * questions, facts * prev_memory, tf.abs(facts - questions),
                        tf.abs(facts - prev_memory)]

            #dim: [max_sentences_per_input, batch_size, 4*hidden_size]
            feature_vectors = tf.concat(2, features)

            #term 1( with W1) = list of max_sentences_per_input: [batch_size, hidden_size]
            #whole expression i.e. with term 2( with W2) = list of max_sentences_per_input:  [batch_size,1]

            """
            #code using einsum (without looping)
            #Not working due to the bug: https://github.com/tensorflow/tensorflow/issues/6384
            #Fixed in version 1.0 onwards!
            inner_term = tf.tanh(matmul_3d_2d(feature_vectors, self.W_1, self.b_1))
            attentions = matmul_3d_2d(inner_term, self.W_2, self.b_2)
            """

            #code using looping
            max_sentences_per_input = feature_vectors.get_shape().as_list()[0]
            attentions = []
            for i in range(max_sentences_per_input):
                attentions.append(tf.matmul(tf.tanh(tf.matmul(feature_vectors[i], self.W_1) + self.b_1), self.W_2) + self.b_2)

            # attentions out: max_sentences_per_input, batch_size
            attentions = tf.squeeze(attentions, axis=2)

            #remove attn
            attn_mask = tf.transpose(tf.sequence_mask(sentences_per_input_in_batch, dtype=tf.float32, maxlen=max_sentences_per_input))
            attentions = attentions * attn_mask

            """
            softmax_att: max_sentences_per_input, batch_size
            dim=0 is needed so we don't need to transpose the attention matrix before applying softmax
            """
            softmax_attn = tf.nn.softmax(attentions, dim=0)

        return softmax_attn

