<<<<<<< HEAD
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2020.3.19

@author: llq
@function:
    1. multi-head attention in theano.op operation
"""
import theano
import theano.tensor as T
import numpy as np
import lasagne
from theano.tests import unittest_tools as utt
from theano import config


def theano_split(x, nums):
    """
    transform numpy split to theano split
    :param x: (batch_size, T, embedding_len)
    :param nums:
    :return: x: (batch_size * h, T, embedding_len / h)
    """
    emb_len = x.shape[-1]

    # (h*N, T_q, d_model/h)
    x = T.concatenate(T.split(x, [emb_len / nums] * nums, nums, axis=2), axis=0)
    return x

def theano_split_restore(x, nums):
    """
    transform numpy split to theano split
    :param x: (batch_size * h, keywords_num, gru_size / h)
    :param nums:
    :return: x: (batch_size, keywords_num, gru_size)
    """
    batch_size = x.shape[0]

    # (batch_size, T, gru_size)
    x = T.concatenate(T.split(x, [batch_size / nums] * nums, nums, axis=0), axis=2)
    return x

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
    outputs, updates = theano.scan(lambda i, x, y: T.dot(x[i], T.transpose(y[i])), outputs_info=None,
                                   sequences=T.arange(K.shape[0]),
                                   non_sequences=[K, V])

    # scale
    outputs /= d_k ** 0.5

    return outputs


class test2(theano.Op):
    __props__ = ()

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        # Note: using x_.type() is dangerous, as it copies x's broadcasting
        # behaviour
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = np.power(x, 2)

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        return [output_grads[0] * 2 * inputs[0]]

    # def R_op(self, inputs, eval_points):
    #     # R_op can receive None as eval_points.
    #     # That mean there is no diferientiable path through that input
    #     # If this imply that you cannot compute some outputs,
    #     # return None for those.
    #     if eval_points[0] is None:
    #         return eval_points
    #     return self.grad(inputs, eval_points)

class Theano_Matmul(T.Op):
    __props__ = ()

    itypes=[T.ftensor3, T.ftensor3]
    otypes=[T.ftensor3]

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        y = inputs[1]

        z = output_storage[0]
        z[0] = np.matmul(x, y)


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
        l_aux_merge = inputs[3]

        w_aux = theano_split(w_aux, self.num_heads)
        w_main = theano_split(w_main, self.num_heads)

        output = scaled_dot_product_attention(w_aux, w_main) # (batch_size * h, keywords_num, max_length)

        # softmax:
        outputs = T.nnet.softmax(T.reshape(output, [-1, output.shape[-1]]))

        # l_H_add = l_merge + l_aux_merge: (batch_size, max_length, gru_size)
        l_H_add, updates = theano.scan(lambda i, x, y: x[i] + T.reshape(y[i], (1, -1)), outputs_info=None,
                                       sequences=T.arange(l_merge.shape[0]),
                                       non_sequences=[l_merge, l_aux_merge])

        # (batch_size * h, max_length, gru_size / h)
        l_H_add = theano_split(l_H_add, self.num_heads)

        # weighted sum (context vectors): (batch_size * h, keywords_num, gru_size / h)
        outputs = T.reshape(outputs, [-1, output.shape[1], output.shape[2]])
        outputs, updates = theano.scan(lambda i, x, y: T.dot(x[i], y[i]), outputs_info = None,
                                        sequences = T.arange(outputs.shape[0]),
                                        non_sequences = [outputs, l_H_add])

        # Restore shape： (batch_size, keywords_num, gru_size)
        outputs = theano_split_restore(outputs, self.num_heads)

        outputs = T.cast(T.sum(outputs, axis=1), "float32")

        return outputs

    def get_output_shape_for(self, input_shapes):
        shapes_fir = input_shapes[0]
        shapes_third = input_shapes[2]
        return (shapes_fir[0], shapes_third[2])


class multihead_attention_origin(lasagne.layers.MergeLayer):
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
        super(multihead_attention_origin, self).__init__(incomings, **kwargs)
        self.num_heads = num_heads

    def get_output_for(self, inputs, **kwargs):
        Q = inputs[0]
        K = inputs[1]
        V = inputs[2]

        # (batch_size * h, max_length, gru_size / h)
        Q = theano_split(Q, self.num_heads)
        K = theano_split(K, self.num_heads)
        V = theano_split(V, self.num_heads)

        # (batch_size * h, max_length, max_length)
        output = scaled_dot_product_attention(Q, K)

        # softmax:
        outputs = T.nnet.softmax(T.reshape(output, [-1, output.shape[-1]]))

        # weighted sum (context vectors): (batch_size * h, max_length, gru_size / h)
        outputs = T.reshape(outputs, [-1, output.shape[1], output.shape[2]])
        outputs, updates = theano.scan(lambda i, x, y: T.dot(x[i], y[i]), outputs_info = None,
                                        sequences = T.arange(outputs.shape[0]),
                                        non_sequences = [outputs, V])

        # Restore shape： (batch_size, max_length, gru_size)
        outputs = theano_split_restore(outputs, self.num_heads)

        return outputs

    def get_output_shape_for(self, input_shapes):
        shapes_fir = input_shapes[0]
        shapes_sec = input_shapes[1]
        shapes_third = input_shapes[2]
        return (shapes_fir[0], shapes_sec[1], shapes_third[2])


def test(w_aux, w_main, l_merge, l_aux_merge, num_heads):
    w_aux = theano_split(w_aux, num_heads)
    w_main = theano_split(w_main, num_heads)

    output = scaled_dot_product_attention(w_aux, w_main)  # (batch_size * h, keywords_num, max_length)

    # softmax:
    outputs = T.nnet.softmax(T.reshape(output, [-1, output.shape[-1]]))

    # l_H_add = l_merge + l_aux_merge: (batch_size, max_length, gru_size)
    l_H_add, updates = theano.scan(lambda i, x, y: x[i] + T.reshape(y[i], (1, -1)), outputs_info=None,
                                   sequences=T.arange(l_merge.shape[0]),
                                   non_sequences=[l_merge, l_aux_merge])

    # (batch_size * h, max_length, gru_size / h)
    l_H_add = theano_split(l_H_add, num_heads)

    # weighted sum (context vectors): (batch_size * h, keywords_num, gru_size / h)
    outputs = T.reshape(outputs, [-1, output.shape[1], output.shape[2]])
    outputs, updates = theano.scan(lambda i, x, y: T.dot(x[i], y[i]), outputs_info=None,
                                   sequences=T.arange(outputs.shape[0]),
                                   non_sequences=[outputs, l_H_add])

    # Restore shape
    outputs = theano_split_restore(outputs, num_heads)

    outputs = T.mean(T.mean(T.mean(outputs, axis=2), axis=1), axis=0)
    return outputs


if __name__ == "__main__":
    features1 = T.ftensor3("features1")
    features2 = T.ftensor3("features2")
    features3 = T.ftensor3("features3")
    features4 = T.fmatrix("features4")
    nums = 5

    output_scaled = test(features1, features2, features3, features4, nums)
    output_grad = T.grad(output_scaled, features1)
    train_fn = theano.function([features1, features2, features3, features4], [output_grad], on_unused_input='warn')

    features1_in = np.float32(np.random.uniform(size=[100, 3, 340]))
    features2_in = np.float32(np.random.uniform(size=[100, 102, 340]))
    features3_in = np.float32(np.random.uniform(size=[100, 102, 250]))
    features4_in = np.float32(np.random.uniform(size=[100, 250]))
    output_cur = train_fn(features1_in, features2_in, features3_in, features4_in)
    print(np.array(output_cur))
=======
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2020.3.19

@author: llq
@function:
    1. multi-head attention in theano.op operation
"""
import theano
import theano.tensor as T
import numpy as np
import lasagne
from theano.tests import unittest_tools as utt
from theano import config


def theano_split(x, nums):
    """
    transform numpy split to theano split
    :param x: (batch_size, T, embedding_len)
    :param nums:
    :return: x: (batch_size * h, T, embedding_len / h)
    """
    emb_len = x.shape[-1]

    # (h*N, T_q, d_model/h)
    x = T.concatenate(T.split(x, [emb_len / nums] * nums, nums, axis=2), axis=0)
    return x

def theano_split_restore(x, nums):
    """
    transform numpy split to theano split
    :param x: (batch_size * h, keywords_num, gru_size / h)
    :param nums:
    :return: x: (batch_size, keywords_num, gru_size)
    """
    batch_size = x.shape[0]

    # (batch_size, T, gru_size)
    x = T.concatenate(T.split(x, [batch_size / nums] * nums, nums, axis=0), axis=2)
    return x

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
    outputs, updates = theano.scan(lambda i, x, y: T.dot(x[i], T.transpose(y[i])), outputs_info=None,
                                   sequences=T.arange(K.shape[0]),
                                   non_sequences=[K, V])

    # scale
    outputs /= d_k ** 0.5

    return outputs


class test2(theano.Op):
    __props__ = ()

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        # Note: using x_.type() is dangerous, as it copies x's broadcasting
        # behaviour
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = np.power(x, 2)

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        return [output_grads[0] * 2 * inputs[0]]

    # def R_op(self, inputs, eval_points):
    #     # R_op can receive None as eval_points.
    #     # That mean there is no diferientiable path through that input
    #     # If this imply that you cannot compute some outputs,
    #     # return None for those.
    #     if eval_points[0] is None:
    #         return eval_points
    #     return self.grad(inputs, eval_points)

class Theano_Matmul(T.Op):
    __props__ = ()

    itypes=[T.ftensor3, T.ftensor3]
    otypes=[T.ftensor3]

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        y = inputs[1]

        z = output_storage[0]
        z[0] = np.matmul(x, y)


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
        l_aux_merge = inputs[3]

        w_aux = theano_split(w_aux, self.num_heads)
        w_main = theano_split(w_main, self.num_heads)

        output = scaled_dot_product_attention(w_aux, w_main) # (batch_size * h, keywords_num, max_length)

        # softmax:
        outputs = T.nnet.softmax(T.reshape(output, [-1, output.shape[-1]]))

        # l_H_add = l_merge + l_aux_merge: (batch_size, max_length, gru_size)
        l_H_add, updates = theano.scan(lambda i, x, y: x[i] + T.reshape(y[i], (1, -1)), outputs_info=None,
                                       sequences=T.arange(l_merge.shape[0]),
                                       non_sequences=[l_merge, l_aux_merge])

        # (batch_size * h, max_length, gru_size / h)
        l_H_add = theano_split(l_H_add, self.num_heads)

        # weighted sum (context vectors): (batch_size * h, keywords_num, gru_size / h)
        outputs = T.reshape(outputs, [-1, output.shape[1], output.shape[2]])
        outputs, updates = theano.scan(lambda i, x, y: T.dot(x[i], y[i]), outputs_info = None,
                                        sequences = T.arange(outputs.shape[0]),
                                        non_sequences = [outputs, l_H_add])

        # Restore shape： (batch_size, keywords_num, gru_size)
        outputs = theano_split_restore(outputs, self.num_heads)

        outputs = T.cast(T.sum(outputs, axis=1), "float32")

        return outputs

    def get_output_shape_for(self, input_shapes):
        shapes_fir = input_shapes[0]
        shapes_third = input_shapes[2]
        return (shapes_fir[0], shapes_third[2])


class multihead_attention_origin(lasagne.layers.MergeLayer):
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
        super(multihead_attention_origin, self).__init__(incomings, **kwargs)
        self.num_heads = num_heads

    def get_output_for(self, inputs, **kwargs):
        Q = inputs[0]
        K = inputs[1]
        V = inputs[2]

        # (batch_size * h, max_length, gru_size / h)
        Q = theano_split(Q, self.num_heads)
        K = theano_split(K, self.num_heads)
        V = theano_split(V, self.num_heads)

        # (batch_size * h, max_length, max_length)
        output = scaled_dot_product_attention(Q, K)

        # softmax:
        outputs = T.nnet.softmax(T.reshape(output, [-1, output.shape[-1]]))

        # weighted sum (context vectors): (batch_size * h, max_length, gru_size / h)
        outputs = T.reshape(outputs, [-1, output.shape[1], output.shape[2]])
        outputs, updates = theano.scan(lambda i, x, y: T.dot(x[i], y[i]), outputs_info = None,
                                        sequences = T.arange(outputs.shape[0]),
                                        non_sequences = [outputs, V])

        # Restore shape： (batch_size, max_length, gru_size)
        outputs = theano_split_restore(outputs, self.num_heads)

        return outputs

    def get_output_shape_for(self, input_shapes):
        shapes_fir = input_shapes[0]
        shapes_sec = input_shapes[1]
        shapes_third = input_shapes[2]
        return (shapes_fir[0], shapes_sec[1], shapes_third[2])


def test(w_aux, w_main, l_merge, l_aux_merge, num_heads):
    w_aux = theano_split(w_aux, num_heads)
    w_main = theano_split(w_main, num_heads)

    output = scaled_dot_product_attention(w_aux, w_main)  # (batch_size * h, keywords_num, max_length)

    # softmax:
    outputs = T.nnet.softmax(T.reshape(output, [-1, output.shape[-1]]))

    # l_H_add = l_merge + l_aux_merge: (batch_size, max_length, gru_size)
    l_H_add, updates = theano.scan(lambda i, x, y: x[i] + T.reshape(y[i], (1, -1)), outputs_info=None,
                                   sequences=T.arange(l_merge.shape[0]),
                                   non_sequences=[l_merge, l_aux_merge])

    # (batch_size * h, max_length, gru_size / h)
    l_H_add = theano_split(l_H_add, num_heads)

    # weighted sum (context vectors): (batch_size * h, keywords_num, gru_size / h)
    outputs = T.reshape(outputs, [-1, output.shape[1], output.shape[2]])
    outputs, updates = theano.scan(lambda i, x, y: T.dot(x[i], y[i]), outputs_info=None,
                                   sequences=T.arange(outputs.shape[0]),
                                   non_sequences=[outputs, l_H_add])

    # Restore shape
    outputs = theano_split_restore(outputs, num_heads)

    outputs = T.mean(T.mean(T.mean(outputs, axis=2), axis=1), axis=0)
    return outputs


if __name__ == "__main__":
    features1 = T.ftensor3("features1")
    features2 = T.ftensor3("features2")
    features3 = T.ftensor3("features3")
    features4 = T.fmatrix("features4")
    nums = 5

    output_scaled = test(features1, features2, features3, features4, nums)
    output_grad = T.grad(output_scaled, features1)
    train_fn = theano.function([features1, features2, features3, features4], [output_grad], on_unused_input='warn')

    features1_in = np.float32(np.random.uniform(size=[100, 3, 340]))
    features2_in = np.float32(np.random.uniform(size=[100, 102, 340]))
    features3_in = np.float32(np.random.uniform(size=[100, 102, 250]))
    features4_in = np.float32(np.random.uniform(size=[100, 250]))
    output_cur = train_fn(features1_in, features2_in, features3_in, features4_in)
    print(np.array(output_cur))
>>>>>>> 9e9c62ef716e71fd5ce04a1694098a1af0372f05
