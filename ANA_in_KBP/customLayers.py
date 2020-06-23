#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:42:27 2018

@author: llq
"""

import lasagne
import theano.tensor as T
import theano


class Input_keywords_DotLayer(lasagne.layers.MergeLayer):
    """
    Input and keywords Dot
    return:
        (batch_size,sentence_length,keywords_number)
    """

    def get_output_for(self, inputs, **kwargs):
        # get two input
        input_fir = inputs[0]
        input_sec = inputs[1]
        # batch_size
        batch_size = T.arange(input_fir.shape[0])

        output, updates = theano.scan(lambda i, x1, x2: T.dot(x1[i], T.transpose(x2[i])), \
                                      outputs_info=None, \
                                      sequences=batch_size, \
                                      non_sequences=[input_fir, input_sec])

        return output

    def get_output_shape_for(self, input_shapes):
        shapes_fir = input_shapes[0]
        shapes_sec = input_shapes[1]

        output_shape = (shapes_fir[0], shapes_fir[1], shapes_sec[1])
        return output_shape


class concat_attention_layer(lasagne.layers.MergeLayer):
    """
    Input:
        1.l_in_dot:(batch_size,max_length,keywords_number)
        2.l_H_dot:(batch_size,max_length,keywords_number)
    return:

    """

    def __init__(self, incomings, w1=lasagne.init.Normal(), b=lasagne.init.Normal(), w2=lasagne.init.Normal(),
                 atten_size=100, **kwargs):
        super(concat_attention_layer, self).__init__(incomings, **kwargs)
        self.keywords_number = self.input_shapes[0][2]
        self.w1 = self.add_param(w1, (self.keywords_number, atten_size), name='w1')
        self.b = self.add_param(b, (atten_size,), name='b')
        self.w2 = self.add_param(w2, (atten_size,), name='w2')

    def get_output_for(self, inputs, **kwargs):
        l_in_dot = inputs[0]
        l_H_dot = inputs[1]
        batch_size = l_in_dot.shape[0]

        # v=tanh(l_in_dot * w2 + b)
        v = T.tanh(T.dot(T.reshape(l_in_dot, (-1, self.keywords_number)), self.w1) + T.reshape(self.b, (1, -1)))
        # alpha=softmax(v * w2) :(batch_size,max_length)
        alphas = T.nnet.softmax(T.reshape(T.dot(v, T.reshape(self.w2, (-1, 1))), (batch_size, -1)))

        # r=alpha * l_H_dot:(batch_size,1,keywords_number)
        r, updates = theano.scan(lambda i, x, y: T.dot(T.reshape(y[i], (1, -1)), x[i]), \
                                 outputs_info=None, \
                                 sequences=T.arange(batch_size), \
                                 non_sequences=[l_H_dot, alphas])

        r = T.reshape(r, (-1, self.keywords_number))
        return r

    def get_output_shape_for(self, input_shapes):
        shapes_fir = input_shapes[0]
        return (shapes_fir[0], shapes_fir[2])


class concat_attention_layer2(lasagne.layers.MergeLayer):
    """
    Input:
        1.l_in_dot:(batch_size,max_length,keywords_number)
        2.l_H_dot:(batch_size,max_length,keywords_number)
    return:

    """

    def __init__(self, incomings, w1=lasagne.init.Normal(), b=lasagne.init.Normal(), w2=lasagne.init.Normal(),
                 atten_size=100, **kwargs):
        super(concat_attention_layer2, self).__init__(incomings, **kwargs)
        #        self.keywords_number = self.input_shapes[0][2]
        self.keywords_number = 107
        self.w1 = self.add_param(w1, (self.keywords_number, atten_size), name='w1')
        self.b = self.add_param(b, (atten_size,), name='b')
        self.w2 = self.add_param(w2, (atten_size,), name='w2')

    def get_output_for(self, inputs, **kwargs):
        l_in_dot = inputs[0]
        l_H_dot = inputs[1]
        batch_size = l_in_dot.shape[0]

        # l_H_dot * l_in_dot^T:(batch_size,max_length,max_length)
        l_in_dot2, updates = theano.scan(lambda i, x, y: T.dot(y[i], T.transpose(x[i])), \
                                         outputs_info=None, \
                                         sequences=T.arange(batch_size), \
                                         non_sequences=[l_in_dot, l_H_dot])

        # v=tanh(l_in_dot * w2 + b)
        v = T.tanh(T.dot(T.reshape(l_in_dot2, (-1, self.keywords_number)), self.w1) + T.reshape(self.b, (1, -1)))
        # alpha=softmax(v * w2) :(batch_size,max_length)
        alphas = T.nnet.softmax(T.reshape(T.dot(v, T.reshape(self.w2, (-1, 1))), (batch_size, -1)))

        # r=alpha * l_H_dot:(batch_size,1,key_words_number)
        r, updates = theano.scan(lambda i, x, y: T.dot(T.reshape(y[i], (1, -1)), x[i]), \
                                 outputs_info=None, \
                                 sequences=T.arange(batch_size), \
                                 non_sequences=[l_H_dot, alphas])

        r = T.reshape(r, (-1, 3))
        return r

    def get_output_shape_for(self, input_shapes):
        shapes_fir = input_shapes[0]
        return (shapes_fir[0], shapes_fir[2])


class concat_attention_layer3(lasagne.layers.MergeLayer):
    """
    Input:
        1.l_in_dot:(batch_size,max_length,keywords_number)
        2.l_merge:(batch_size,max_length,gru_size)
        3.l_aux_merge:(batch_size,gru_size)

    Add the "l_merge" and "l_aux_merge"

    """

    def __init__(self, incomings, w1=lasagne.init.Normal(), b=lasagne.init.Normal(), w2=lasagne.init.Normal(),
                 atten_size=100, **kwargs):
        super(concat_attention_layer3, self).__init__(incomings, **kwargs)
        self.keywords_number = self.input_shapes[0][2]
        self.w1 = self.add_param(w1, (self.keywords_number, atten_size), name='w1')
        self.b = self.add_param(b, (atten_size,), name='b')
        self.w2 = self.add_param(w2, (atten_size,), name='w2')

    def get_output_for(self, inputs, **kwargs):
        l_in_dot = inputs[0]
        l_merge = inputs[1]
        l_aux_merge = inputs[2]
        batch_size = l_in_dot.shape[0]

        # l_H_add = l_merge + l_aux_merge:(batch_size,max_length,gru_size)
        l_H_add, updates = theano.scan(lambda i, x, y: x[i] + T.reshape(y[i], (1, -1)), \
                                       outputs_info=None, \
                                       sequences=T.arange(batch_size), \
                                       non_sequences=[l_merge, l_aux_merge])

        # v=tanh(l_in_dot * w1 + b)
        v = T.tanh(T.dot(T.reshape(l_in_dot, (-1, self.keywords_number)), self.w1) + T.reshape(self.b, (1, -1)))
        # alpha=softmax(v * w2) :(batch_size,max_length)
        alphas = T.nnet.softmax(T.reshape(T.dot(v, T.reshape(self.w2, (-1, 1))), (batch_size, -1)))

        # r=alpha * l_H_dot:(batch_size,1,gru_size)
        r, updates = theano.scan(lambda i, x, y: T.dot(T.reshape(y[i], (1, -1)), x[i]), \
                                 outputs_info=None, \
                                 sequences=T.arange(batch_size), \
                                 non_sequences=[l_H_add, alphas])

        r = T.reshape(r, (batch_size, -1))
        return r

    def get_output_shape_for(self, input_shapes):
        shapes_sec = input_shapes[1]
        return (shapes_sec[0], shapes_sec[2])


class concat_attention_layer3_1(lasagne.layers.MergeLayer):
    """
    Input:
        1.l_in_dot:(batch_size,max_length,keywords_number)
        2.l_merge:(batch_size,max_length,gru_size)
        3.l_aux_merge:(batch_size,gru_size)

    Add the "l_merge" and "l_aux_merge"

    """

    def __init__(self, incomings, **kwargs):
        super(concat_attention_layer3_1, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        l_in_dot = inputs[0]
        l_merge = inputs[1]
        l_aux_merge = inputs[2]
        batch_size = l_in_dot.shape[0]

        # l_H_add = l_merge + l_aux_merge:(batch_size,max_length,gru_size)
        l_H_add, updates = theano.scan(lambda i, x, y: x[i] + T.reshape(y[i], (1, -1)), \
                                       outputs_info=None, \
                                       sequences=T.arange(batch_size), \
                                       non_sequences=[l_merge, l_aux_merge])

        return l_H_add

    def get_output_shape_for(self, input_shapes):
        shapes_sec = input_shapes[1]
        return (shapes_sec[0], shapes_sec[1], shapes_sec[2])


class concat_attention_layer3_2(lasagne.layers.MergeLayer):
    """
    Input:
        1.l_in_dot:(batch_size,max_length,keywords_number)
        2.l_merge:(batch_size,max_length,gru_size)
        3.l_aux_merge:(batch_size,gru_size)

    Add the "l_merge" and "l_aux_merge"

    """

    def __init__(self, incomings, w1=lasagne.init.Normal(), b=lasagne.init.Normal(), w2=lasagne.init.Normal(),
                 atten_size=100, **kwargs):
        super(concat_attention_layer3_2, self).__init__(incomings, **kwargs)
        self.keywords_number = self.input_shapes[0][2]
        self.w1 = self.add_param(w1, (self.keywords_number, atten_size), name='w1')
        self.b = self.add_param(b, (atten_size,), name='b')
        self.w2 = self.add_param(w2, (atten_size,), name='w2')

    def get_output_for(self, inputs, **kwargs):
        l_in_dot = inputs[0]
        l_merge = inputs[1]
        l_aux_merge = inputs[2]
        batch_size = l_in_dot.shape[0]

        # l_H_add = l_merge + l_aux_merge:(batch_size,max_length,gru_size)
        l_H_add, updates = theano.scan(lambda i, x, y: x[i] + T.reshape(y[i], (1, -1)), \
                                       outputs_info=None, \
                                       sequences=T.arange(batch_size), \
                                       non_sequences=[l_merge, l_aux_merge])

        # v=tanh(l_in_dot * w1 + b)
        v = T.tanh(T.dot(T.reshape(l_in_dot, (-1, self.keywords_number)), self.w1) + T.reshape(self.b, (1, -1)))
        # alpha=softmax(v * w2) :(batch_size,max_length)
        alphas = T.nnet.softmax(T.reshape(T.dot(v, T.reshape(self.w2, (-1, 1))), (batch_size, -1)))

        return alphas

    def get_output_shape_for(self, input_shapes):
        shapes_sec = input_shapes[1]
        return (shapes_sec[0], shapes_sec[1])


class concat_attention_layer3_3(lasagne.layers.MergeLayer):
    """
    Input:
        1.l_in_dot:(batch_size,max_length,keywords_number)
        2.l_merge:(batch_size,max_length,gru_size)
        3.l_aux_merge:(batch_size,gru_size)

    Add the "l_merge" and "l_aux_merge"

    """

    def __init__(self, incomings, **kwargs):
        super(concat_attention_layer3_3, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        l_H_add = inputs[0]
        alphas = inputs[1]
        batch_size = l_H_add.shape[0]

        # r=alpha * l_H_dot:(batch_size,1,gru_size)
        r, updates = theano.scan(lambda i, x, y: T.dot(T.reshape(y[i], (1, -1)), x[i]), \
                                 outputs_info=None, \
                                 sequences=T.arange(batch_size), \
                                 non_sequences=[l_H_add, alphas])

        r = T.reshape(r, (batch_size, -1))
        return r

    def get_output_shape_for(self, input_shapes):
        shapes_sec = input_shapes[0]
        return (shapes_sec[0], shapes_sec[2])