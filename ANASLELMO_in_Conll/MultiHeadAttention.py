#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2020.3.19

@author: llq
@function:
    1. multi-head attention
"""
import lasagne
import numpy as np
import theano.tensor as T
import theano
from theano.compile.ops import as_op


@as_op(itypes=[T.ftensor3, T.iscalar],
       otypes=[T.ftensor3])
def theano_split(x, nums):
    """
    transform numpy split to theano split
    :param x: (batch_size, T, embedding_len)
    :param nums:
    :return: x: (batch_size * h, T, embedding_len / h)
    """
    # (h*N, T_q, d_model/h)
    x = np.concatenate(np.split(x, nums, axis=2), axis=0)

    return x

@as_op(itypes=[T.ftensor3, T.iscalar],
       otypes=[T.ftensor3])
def theano_split_restore(x, nums):
    """
    transform numpy split to theano split
    :param x: (batch_size * h, T, gru_size)
    :param nums:
    :return: x:
    """
    # (h*N, T_q, d_model/h)
    x = np.concatenate(np.split(x, nums, axis=0), axis=2)

    return x

@as_op(itypes=[T.ftensor3, T.ftensor3],
       otypes=[T.ftensor3])
def scaled_dot_product_attention(K, V):
    """
    scaled operation
    :param K: (batch_size * h, keywords_num, embedding_len / h) -> w_aux
    :param V: (batch_size * h, max_length, embedding_len / h) -> w_main
    :return:
        output: (batch_size * h, keywords_num, max_length)
    """
    d_k = K.shape[-1]

    # dot product: (batch_size * h, keywords_num, max_length)
    outputs = np.matmul(K, V.transpose([0, 2, 1]))

    # scale
    outputs /= d_k ** 0.5

    return outputs


@as_op(itypes=[T.ftensor3, T.ftensor3],
       otypes=[T.ftensor3])
def theano_matmul(x, y):
    output = np.matmul(x, y)
    return output


class multihead_attention(lasagne.layers.MergeLayer):
    """
    Function:
        realise multi-head attention
    Input:
        1. w_aux: (batch_size, keywords_num, embedding_len)
        2. w_main: (batch_size, max_length, embedding_len)
        3. l_merge: (batch_size, max_length, gru_size)
        4. l_aux_merge: (batch_size, gru_size)
    Reutrn:
        1. output: (batch_size, keywords_num, gru_size)
    """

    def __init__(self, incomings, num_heads=5, **kwargs):
        super(multihead_attention, self).__init__(incomings, **kwargs)
        self.num_heads = num_heads

    def get_output_for(self, inputs, **kwargs):
        w_aux = inputs[0]
        w_main = inputs[1]
        l_merge = inputs[2]
        l_aux_merge = inputs[4]

        w_aux = theano_split(w_aux, self.num_heads)
        w_main = theano_split(w_main, self.num_heads)

        output = scaled_dot_product_attention(w_aux, w_main) # (batch_size * h, keywords_num, max_length)

        # softmax:
        outputs = T.nnet.softmax(T.reshape(output, [-1, output.shape[-1]]))

        # l_H_add = l_merge + l_aux_merge: (batch_size, max_length, gru_size)
        l_H_add, updates = theano.scan(lambda i, x, y: x[i] + T.reshape(y[i], (1, -1)), \
                                       outputs_info=None, \
                                       sequences=T.arange(l_merge.shape[0]), \
                                       non_sequences=[l_merge, l_aux_merge])

        # (batch_size * h, max_length, gru_size / h)
        l_H_add = theano_split(l_H_add, self.num_heads)

        # weighted sum (context vectors): (batch_size * h, keywords_num, gru_size / h)
        outputs = theano_matmul(T.reshape(output, [-1, output.shape[1], output.shape[2]]), l_H_add)

        # Restore shape
        outputs = theano_split_restore(outputs, self.num_heads)

        return outputs

    def get_output_shape_for(self, input_shapes):
        shapes_fir = input_shapes[0]
        shapes_third = input_shapes[2]
        return (shapes_fir[0], shapes_fir[1], shapes_third[2])


def test(w_aux, w_main, l_merge, l_aux_merge, num_heads):
    w_aux = theano_split(w_aux, num_heads)
    w_main = theano_split(w_main, num_heads)

    output = scaled_dot_product_attention(w_aux, w_main)  # (batch_size * h, keywords_num, max_length)

    # softmax:
    outputs = T.nnet.softmax(T.reshape(output, [-1, output.shape[-1]]))

    # l_H_add = l_merge + l_aux_merge: (batch_size, max_length, gru_size)
    l_H_add, updates = theano.scan(lambda i, x, y: x[i] + T.reshape(y[i], (1, -1)),
                                   outputs_info=None,
                                   sequences=T.arange(l_merge.shape[0]),
                                   non_sequences=[l_merge, l_aux_merge])

    # (batch_size * h, max_length, gru_size / h)
    l_H_add = theano_split(l_H_add, num_heads)

    # weighted sum (context vectors): (batch_size * h, keywords_num, gru_size / h)
    outputs = theano_matmul(T.reshape(output, [-1, output.shape[1], output.shape[2]]), l_H_add)

    # Restore shape
    outputs = theano_split_restore(outputs, num_heads)

    return outputs


if __name__ == "__main__":
    features1 = T.ftensor3("features1")
    features2 = T.ftensor3("features2")
    features3 = T.ftensor3("features3")
    features4 = T.fmatrix("features4")
    num_heads = T.iscalar("num_heads")

    output_scaled = test(features1, features2, features3, features4, num_heads)
    train_fn = theano.function([features1, features2, features3, features4, num_heads], [output_scaled], on_unused_input='warn')

    features1_in = np.float32(np.random.uniform(size=[100, 3, 340]))
    features2_in = np.float32(np.random.uniform(size=[100, 102, 340]))
    features3_in = np.float32(np.random.uniform(size=[100, 102, 250]))
    features4_in = np.float32(np.random.uniform(size=[100, 250]))
    output_cur = train_fn(features1_in, features2_in, features3_in, features4_in, 5)
    print(np.array(output_cur).shape)
