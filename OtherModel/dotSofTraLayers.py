#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 08:34:13 2017

crete lasagne.layers.Layer of Dot,Softmax,Transpose,Attention

@author: llq
"""
import lasagne
import theano.tensor as T
import theano


class DotMatrixLayer(lasagne.layers.MergeLayer):
    """
    Two maxtric dot.
    Multiplied by the corresponding matrix(2D) of the page.
    """

    def get_output_for(self, inputs, **kwargs):
        """
        1.Return the maxtric dot.
            [0]:
                2D*2D
            [1]:
                2D*2D
            ...
        2.sum:
            sum the "attention_size"

        3.return:(batch_size,gru_size)
        """
        # get two input
        input_fir = inputs[0]
        input_sec = inputs[1]
        # batch_size
        batch_size = T.arange(input_fir.shape[0])

        output, updates = theano.scan(lambda i, x1, x2: T.dot(x1[i], x2[i]), \
                                      outputs_info=None, \
                                      sequences=batch_size, \
                                      non_sequences=[input_fir, input_sec])

        # sum the attention_size
        output = T.sum(output, axis=1)
        return output

    def get_output_shape_for(self, input_shapes):
        shapes_fir = input_shapes[0]
        shapes_sec = input_shapes[1]

        output_shape = (shapes_fir[0], shapes_sec[2])
        return output_shape


class SoftmaxMatrixLayer(lasagne.layers.Layer):
    """
    Softmax the 2D matrix
    """

    def get_output_for(self, input, **kwargs):
        """
        Softmax the matrix:
            [0]:
                softmax(matrix1)
            [1]:
                softmax(matrix2)
            ...
        """
        # batch_size
        batch_size = T.arange(input.shape[0])

        output, updates = theano.scan(lambda i, x: T.nnet.nnet.softmax(x[i]), \
                                      outputs_info=None, \
                                      sequences=batch_size, \
                                      non_sequences=input)

        return output


class TransposeMatrixLayer(lasagne.layers.Layer):
    """
    Transpose the 2D matrix.
    """

    def get_output_for(self, input, **kwargs):
        """
        Transpose the matrix:
            [0]:
                transpose(matrix1)
            [1]:
                transpose(matrix2)
            ...

            return:(batch_size,attention_size,num_steps)
        """
        # batch_size
        batch_size = T.arange(input.shape[0])

        output, updates = theano.scan(lambda i, x: T.transpose(x[i]), \
                                      outputs_info=None, \
                                      sequences=batch_size, \
                                      non_sequences=input)

        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[1])


class AttentionLayer(lasagne.layers.Layer):
    """
    Attention layer:
        tanh(H)*(attention_w^T)

        return:(batch_size,num_steps,attention_size)
    """

    def __init__(self, incoming, attention_w=lasagne.init.Normal(), attention_b=lasagne.init.Normal(),
                 attention_u=lasagne.init.Normal(), atten_size=100, **kwargs):
        super(AttentionLayer, self).__init__(incoming, **kwargs)
        #        num_steps = self.input_shape[1]
        gru_size = self.input_shape[2]
        self.attention_w = self.add_param(attention_w, (gru_size, atten_size), name='attention_w')
        self.attention_b = self.add_param(attention_b, (atten_size,), name='attention_b')
        self.attention_u = self.add_param(attention_u, (atten_size,), name='attention_u')

    def get_output_for(self, input, **kwargs):
        # num_steps:time steps
        num_steps = input.shape[1]
        # gru_size
        gru_size = input.shape[2]

        # v=tanh(w*input+b):(batch_size * num_steps,attention_size)
        v = T.tanh(T.dot(T.reshape(input, (-1, gru_size)), self.attention_w) + T.reshape(self.attention_b, (1, -1)))
        # e=v*u:(batch_size * num_steps)
        vu = T.dot(v, T.reshape(self.attention_u, (-1, 1)))
        # alphas=softmax(e):
        #    exps:(batch_size,num_steps)
        exps = T.reshape(T.exp(vu), (-1, num_steps))
        alphas = exps / T.reshape(T.sum(exps, axis=1), (-1, 1))

        # s=alphas*H:(batch_size,gru_size)
        s = T.sum(input * T.reshape(alphas, (-1, num_steps, 1)), 1)

        # (batch_size,gru_size)
        return s

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])


class InputAttentionLayer(lasagne.layers.Layer):
    """
    Attention layer:
        v=l_in*l_input_att
        alpha=softmax(v)
        r=alpha*l_in
        return:(batch_size,num_steps,attention_size)
    """

    def __init__(self, incoming, attention_w=lasagne.init.Normal(), attention_b=lasagne.init.Normal(),
                 attention_u=lasagne.init.Normal(), atten_size=100, **kwargs):
        super(InputAttentionLayer, self).__init__(incoming, **kwargs)
        #        num_steps = self.input_shape[1]
        vec_len = self.input_shape[2]
        self.atten_size = atten_size
        self.attention_w = self.add_param(attention_w, (vec_len, atten_size), name='attention_w')
        self.attention_b = self.add_param(attention_b, (atten_size,), name='attention_b')
        self.attention_u = self.add_param(attention_u, (atten_size,), name='attention_u')

    def get_output_for(self, input, **kwargs):
        l_in = input
        #        l_input_att=inputs[1]

        # num_steps:time steps
        num_steps = l_in.shape[1]
        # vector length
        vec_len = l_in.shape[2]

        # v=tanh(w*input+b):(batch_size * num_steps,attention_size)
        v = T.tanh(T.dot(T.reshape(l_in, (-1, vec_len)), self.attention_w) + T.reshape(self.attention_b, (1, -1)))
        # e=v*u:(batch_size * num_steps)
        vu = T.dot(v, T.reshape(self.attention_u, (-1, 1)))
        # alphas=softmax(e):
        #    exps:(batch_size,num_steps)
        exps = T.reshape(T.exp(vu), (-1, num_steps))
        alphas = exps / T.reshape(T.sum(exps, axis=1), (-1, 1))

        # s=alphas*H:(batch_size,gru_size)
        s = l_in * T.reshape(alphas, (-1, num_steps, 1))

        '''
        #v=dot(l_in,l_input_att):(batch_size,1,num_steps)
        l_in=T.tanh(l_in)
        v,updates=theano.scan(lambda i,x,y:T.dot(T.reshape(y[i],(1,-1)),T.transpose(x[i])),\
           outputs_info=None,\
           sequences=T.arange(l_in.shape[0]),\
           non_sequences=[l_in,l_input_att])

        #v:(batch_size,num_steps)
        v=T.reshape(v,(-1,num_steps))

        #alphas=softmax(v):
        #    exps:(batch_size,num_steps)  
        exps = T.exp(v)
        alphas = exps / T.reshape(T.sum(exps, axis=1), (-1, 1))

        #r=alpha*l_in
        r,updates=theano.scan(lambda i,x,y:T.reshape(x[i],(-1,1))*y[i],\
           outputs_info=None,\
           sequences=T.arange(alphas.shape[0]),\
           non_sequences=[alphas,l_in])
        '''

        '''
        #v=tanh(w*input+b):(batch_size, num_steps, atten_size)
#        v=T.tanh(T.dot(T.reshape(l_in,(-1,vec_len)),self.attention_w)+T.reshape(self.attention_b,(1,-1)))
#        v=T.reshape(v,(-1,num_steps,self.atten_size))
        v=T.tanh(l_in)

        #e=v*l_input_att:(batch_size,num_steps)
        e,updates=theano.scan(lambda i,x,y:T.dot(x[i],y[i]),\
           outputs_info=None,\
           sequences=T.arange(l_in.shape[0]),\
           non_sequences=[v,l_input_att])

        #alphas=softmax(e):
        #    exps:(batch_size,num_steps) 
        exps = T.reshape(T.exp(e), (-1, num_steps))
        alphas = exps / T.reshape(T.sum(exps, axis=1), (-1, 1))

        #r=alphas*l_in:(batch_size,gru_size)
        r=T.sum(l_in * T.reshape(alphas, (-1, num_steps, 1)), 1)
        '''
        return s

    def get_output_shape_for(self, input_shape):
        shapes_fir = input_shape
        return (shapes_fir[0], shapes_fir[1], shapes_fir[2])


class InputAttEntRootLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, u_e1=lasagne.init.Normal(), u_e2=lasagne.init.Normal(), u_root=lasagne.init.Normal(),
                 ws2=lasagne.init.Normal(), **kwargs):
        super(InputAttEntRootLayer, self).__init__(incomings, **kwargs)
        #        batch_size=100
        vec_len = self.input_shapes[0][2]
        self.u_e1 = self.add_param(u_e1, (vec_len,), name='u_e1')
        self.u_e2 = self.add_param(u_e2, (vec_len,), name='u_e2')
        self.u_root = self.add_param(u_root, (vec_len,), name='u_root')
        self.attention_size = 10
        self.ws2 = self.add_param(ws2, (3, self.attention_size), name='ws2')

    def get_output_for(self, inputs, **kwargs):
        # input
        l_in = inputs[0]
        # entity e1 embedding
        l_entity_e1 = inputs[1] * self.u_e1
        # entity e2 embedding
        l_entity_e2 = inputs[2] * self.u_e2
        # root embedding
        l_root = inputs[3] * self.u_root
        #        #entity e1 embedding
        #        l_entity_e1=self.u_e1
        #        #entity e2 embedding
        #        l_entity_e2=self.u_e2
        #        #root embedding
        #        l_root=self.u_root

        # num_steps
        num_steps = l_in.shape[1]
        # vector length
        vec_len = l_in.shape[2]

        """alphas_e1"""
        # e1_a=e*w:(batch_size*num_steps,1)
        l_e1_a, updates = theano.scan(lambda i, x, y: T.dot(T.reshape(x[i], (-1, vec_len)), y[i]), \
                                      outputs_info=None, \
                                      sequences=T.arange(l_in.shape[0]), \
                                      non_sequences=[l_in, l_entity_e1])

        # alphas_e1=softmax(a):(batch_size,num_steps)
        l_e1_a = T.reshape(l_e1_a, (-1, num_steps))
        e1_max = T.reshape(T.max(l_e1_a, axis=1), (-1, 1))
        exps_e1 = T.exp(l_e1_a - e1_max)
        alphas_e1 = exps_e1 / T.reshape(T.sum(exps_e1, axis=1), (-1, 1))

        """alphas_e2"""
        # e1_a=e*w:(batch_size*num_steps,1)
        l_e2_a, updates = theano.scan(lambda i, x, y: T.dot(T.reshape(x[i], (-1, vec_len)), y[i]), \
                                      outputs_info=None, \
                                      sequences=T.arange(l_in.shape[0]), \
                                      non_sequences=[l_in, l_entity_e2])

        # alphas_e2=softmax(a):(batch_size,num_steps)
        l_e2_a = T.reshape(l_e2_a, (-1, num_steps))
        e2_max = T.reshape(T.max(l_e2_a, axis=1), (-1, 1))
        exps_e2 = T.exp(l_e2_a - e2_max)
        alphas_e2 = exps_e2 / T.reshape(T.sum(exps_e2, axis=1), (-1, 1))

        """alphas_root"""
        # e1_a=e*w:(batch_size*num_steps,1)
        l_root_a, updates = theano.scan(lambda i, x, y: T.dot(T.reshape(x[i], (-1, vec_len)), y[i]), \
                                        outputs_info=None, \
                                        sequences=T.arange(l_in.shape[0]), \
                                        non_sequences=[l_in, l_root])

        # alphas_root=softmax(a):(batch_size,num_steps)
        l_root_a = T.reshape(l_root_a, (-1, num_steps))
        root_max = T.reshape(T.max(l_root_a, axis=1), (-1, 1))
        exps_root = T.exp(l_root_a - root_max)
        alphas_root = exps_root / T.reshape(T.sum(exps_root, axis=1), (-1, 1))

        #        #alpha:(batch_size,num_steps)
        #        alphas=(alphas_e1+alphas_e2+alphas_root)/3.0
        # alpha:(batch_size,num_steps,3)
        alphas_e1 = T.reshape(alphas_e1, (-1, num_steps, 1))
        alphas_e2 = T.reshape(alphas_e2, (-1, num_steps, 1))
        alphas_root = T.reshape(alphas_root, (-1, num_steps, 1))
        alphas = T.concatenate((alphas_e1, alphas_e2, alphas_root), axis=2)
        #        alphas=T.reshape(T.dot(T.reshape(alphas,(-1,3)),self.ws2),(-1,num_steps,self.attention_size))

        #        #r=l_in*alpha:(batch_size,num_steps,vec_len)
        #        r,updates=theano.scan(lambda i,x,y:x[i]*T.reshape(y[i],(-1,1)),\
        #           outputs_info=None,\
        #           sequences=T.arange(l_in.shape[0]),\
        #           non_sequences=[l_in,alphas])

        #        r,updates=theano.scan(lambda i,x,y:T.dot(T.transpose(x[i]),T.reshape(y[i],(-1,1))),\
        #           outputs_info=None,\
        #           sequences=T.arange(l_in.shape[0]),\
        #           non_sequences=[l_in,alphas])
        #        r=T.reshape(r,(-1,vec_len))
        return alphas

    def get_output_shape_for(self, input_shapes):
        shapes_fir = input_shapes[0]
        return (shapes_fir[0], shapes_fir[1], 3)


class InputAttentionDotLayer(lasagne.layers.MergeLayer):
    """
    Two maxtric dot in input attention
    """

    def get_output_for(self, inputs, **kwargs):
        # get two input
        input_fir = inputs[0]
        input_alpha = inputs[1]

        # batch_size
        batch_size = input_fir.shape[0]
        # vector length
        vec_len = input_fir.shape[1]

        # input_alpha:(batch_size,vec_len,1)
        input_alpha = T.reshape(input_alpha, (-1, vec_len, 1))

        # r:(batch_size,num_filters,convolution_len)
        r, updates = theano.scan(lambda i, x, y: x[i] * y[i], \
                                 outputs_info=None, \
                                 sequences=T.arange(batch_size), \
                                 non_sequences=[input_fir, input_alpha])
        #        r=T.reshape(r,(-1,num_filters,convolution_len,1))
        r = T.sum(r, axis=1)
        return r

    def get_output_shape_for(self, input_shapes):
        shapes_fir = input_shapes[0]
        output_shape = (shapes_fir[0], shapes_fir[2])
        return output_shape


class SelfAttentionDotLayer(lasagne.layers.MergeLayer):
    """
    Self-attention
    """

    def get_output_for(self, inputs, **kwargs):
        # get two input
        input_fir = inputs[0]
        input_alpha = inputs[1]

        # batch_size
        batch_size = input_fir.shape[0]

        # input_alpha:(batch_size,3,vec_len)
        input_alpha, updates = theano.scan(lambda i, x: T.transpose(x[i]), \
                                           outputs_info=None, \
                                           sequences=T.arange(batch_size), \
                                           non_sequences=[input_alpha])

        # r:(batch_size,3,gru_size)
        r, updates = theano.scan(lambda i, x, y: T.dot(y[i], x[i]), \
                                 outputs_info=None, \
                                 sequences=T.arange(batch_size), \
                                 non_sequences=[input_fir, input_alpha])
        return r

    def get_output_shape_for(self, input_shapes):
        shapes_fir = input_shapes[0]
        shapes_sec = input_shapes[1]
        output_shape = (shapes_fir[0], shapes_sec[2], shapes_fir[2])
        return output_shape


"""
keywords-Attention:main
"""


class SelfAttEntRootLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, u_e1=lasagne.init.Normal(), u_e2=lasagne.init.Normal(), u_root=lasagne.init.Normal(),
                 e1_ws2=lasagne.init.Normal(), e2_ws2=lasagne.init.Normal(), root_ws2=lasagne.init.Normal(),
                 alphas=lasagne.init.Normal(), attention_size2=20, **kwargs):
        super(SelfAttEntRootLayer, self).__init__(incomings, **kwargs)
        self.attention_size1 = 350
        self.u_e1 = self.add_param(u_e1, (1, self.attention_size1), name='u_e1')
        self.u_e2 = self.add_param(u_e2, (1, self.attention_size1), name='u_e2')
        self.u_root = self.add_param(u_root, (1, self.attention_size1), name='u_root')
        self.attention_size2 = attention_size2
        self.e1_ws2 = self.add_param(e1_ws2, (self.attention_size1, self.attention_size2), name='e1_ws2')
        self.e2_ws2 = self.add_param(e2_ws2, (self.attention_size1, self.attention_size2), name='e2_ws2')
        self.root_ws2 = self.add_param(root_ws2, (self.attention_size1, self.attention_size2), name='root_ws2')

    #        self.alphas=self.add_param(alphas, (100,self.attention_size2,self.input_shapes[0][1]), name='alphas')

    def get_output_for(self, inputs, **kwargs):
        # input
        l_in = inputs[0]
        # entity e1 embedding
        l_entity_e1 = inputs[1]
        # entity e2 embedding
        l_entity_e2 = inputs[2]
        # root embedding
        l_root = inputs[3]
        # l_merge
        l_merge = inputs[4]

        # num_steps
        num_steps = l_in.shape[1]

        """alphas_e1"""
        # e1_a=e*w:(batch_size*num_steps,attention_size2)
        l_e1_a, updates = theano.scan(
            lambda i, x, y, z: T.dot(T.tanh(T.dot(T.dot(x[i], T.reshape(y[i], (-1, 1))), z)), self.e1_ws2), \
            outputs_info=None, \
            sequences=T.arange(l_in.shape[0]), \
            non_sequences=[l_in, l_entity_e1, self.u_e1])

        # alphas_e1=softmax(a):(batch_size,attention_size2,num_steps)
        exps_e1 = T.reshape(T.exp(l_e1_a), (-1, num_steps, self.attention_size2))
        exps_e1, updates = theano.scan(lambda i, x: T.transpose(x[i]), \
                                       outputs_info=None, \
                                       sequences=T.arange(exps_e1.shape[0]), \
                                       non_sequences=[exps_e1])
        alphas_e1 = exps_e1 / T.reshape(T.sum(exps_e1, axis=2), (-1, self.attention_size2, 1))

        """alphas_e2"""
        # e1_a=e*w:(batch_size*num_steps,attention_size2)
        l_e2_a, updates = theano.scan(
            lambda i, x, y, z: T.dot(T.tanh(T.dot(T.dot(x[i], T.reshape(y[i], (-1, 1))), z)), self.e2_ws2), \
            outputs_info=None, \
            sequences=T.arange(l_in.shape[0]), \
            non_sequences=[l_in, l_entity_e2, self.u_e2])

        # alphas_e1=softmax(a):(batch_size,attention_size2,num_steps)
        exps_e2 = T.reshape(T.exp(l_e2_a), (-1, num_steps, self.attention_size2))
        exps_e2, updates = theano.scan(lambda i, x: T.transpose(x[i]), \
                                       outputs_info=None, \
                                       sequences=T.arange(exps_e2.shape[0]), \
                                       non_sequences=[exps_e2])
        alphas_e2 = exps_e2 / T.reshape(T.sum(exps_e2, axis=2), (-1, self.attention_size2, 1))

        """alphas_root"""
        # e1_a=e*w:(batch_size*num_steps,attention_size2)
        l_root_a, updates = theano.scan(
            lambda i, x, y, z: T.dot(T.tanh(T.dot(T.dot(x[i], T.reshape(y[i], (-1, 1))), z)), self.root_ws2), \
            outputs_info=None, \
            sequences=T.arange(l_in.shape[0]), \
            non_sequences=[l_in, l_root, self.u_root])

        # alphas_e1=softmax(a):(batch_size,attention_size2,num_steps)
        exps_root = T.reshape(T.exp(l_root_a), (-1, num_steps, self.attention_size2))
        exps_root, updates = theano.scan(lambda i, x: T.transpose(x[i]), \
                                         outputs_info=None, \
                                         sequences=T.arange(exps_root.shape[0]), \
                                         non_sequences=[exps_root])
        alphas_root = exps_root / T.reshape(T.sum(exps_root, axis=2), (-1, self.attention_size2, 1))

        # alpha:(batch_size,attention_size2,num_steps)
        alphas = (alphas_e1 + alphas_e2 + alphas_root) / 3.0

        # r=l_in*alpha:(batch_size,attention_size2,gru_size)
        r, updates = theano.scan(lambda i, x, y: T.dot(y[i], x[i]), \
                                 outputs_info=None, \
                                 sequences=T.arange(l_in.shape[0]), \
                                 non_sequences=[l_merge, alphas])

        #        r_e1,updates=theano.scan(lambda i,x,y:T.dot(y[i],x[i]),\
        #           outputs_info=None,\
        #           sequences=T.arange(l_in.shape[0]),\
        #           non_sequences=[l_merge,alphas_e1])
        #        r_e2,updates=theano.scan(lambda i,x,y:T.dot(y[i],x[i]),\
        #           outputs_info=None,\
        #           sequences=T.arange(l_in.shape[0]),\
        #           non_sequences=[l_merge,alphas_e2])
        #        r_root,updates=theano.scan(lambda i,x,y:T.dot(y[i],x[i]),\
        #           outputs_info=None,\
        #           sequences=T.arange(l_in.shape[0]),\
        #           non_sequences=[l_merge,alphas_root])
        #        r=T.concatenate((r_root,r_e1,r_e2,),axis=2)

        return alphas

    def get_output_shape_for(self, input_shapes):
        shapes_fir = input_shapes[4]
        return (shapes_fir[0], self.attention_size2, shapes_fir[1])


#        shapes_fir=input_shapes[0]
#        return (shapes_fir[0],self.attention_size2,shapes_fir[1])

class SelfAttEntRootLayer_emp(lasagne.layers.MergeLayer):
    """
    dot betweent the alphas and l_merge
    """

    def get_output_for(self, inputs, **kwargs):
        # input
        alphas = inputs[0]
        l_merge = inputs[1]

        # r=l_in*alpha:(batch_size,attention_size2,gru_size)
        r, updates = theano.scan(lambda i, x, y: T.dot(y[i], x[i]), \
                                 outputs_info=None, \
                                 sequences=T.arange(l_merge.shape[0]), \
                                 non_sequences=[l_merge, alphas])

        return r

    def get_output_shape_for(self, input_shapes):
        shapes_alphas = input_shapes[0]
        shapes_fir = input_shapes[1]
        return (shapes_fir[0], shapes_alphas[1], shapes_fir[2])


"""
keywords-Attention:varints
"""


class SelfAttEntRootLayerVariant(lasagne.layers.MergeLayer):

    def __init__(self, incomings, u_e1=lasagne.init.Normal(), u_e2=lasagne.init.Normal(), u_root=lasagne.init.Normal(),
                 e1_ws2=lasagne.init.Normal(), e2_ws2=lasagne.init.Normal(), root_ws2=lasagne.init.Normal(),
                 alphas=lasagne.init.Normal(), attention_size2=20, **kwargs):
        super(SelfAttEntRootLayerVariant, self).__init__(incomings, **kwargs)
        self.attention_size1 = 350
        self.u_e1 = self.add_param(u_e1, (1, self.attention_size1), name='u_e1')
        self.u_e2 = self.add_param(u_e2, (1, self.attention_size1), name='u_e2')
        self.u_root = self.add_param(u_root, (1, self.attention_size1), name='u_root')
        self.attention_size2 = attention_size2
        self.e1_ws2 = self.add_param(e1_ws2, (self.attention_size1, self.attention_size2), name='e1_ws2')
        self.e2_ws2 = self.add_param(e2_ws2, (self.attention_size1, self.attention_size2), name='e2_ws2')
        self.root_ws2 = self.add_param(root_ws2, (self.attention_size1, self.attention_size2), name='root_ws2')

    #        self.alphas=self.add_param(alphas, (100,self.attention_size2,self.input_shapes[0][1]), name='alphas')

    def get_output_for(self, inputs, **kwargs):
        # input
        l_in = inputs[0]
        # entity e1 embedding
        l_entity_e1 = inputs[1]
        # entity e2 embedding
        l_entity_e2 = inputs[2]
        # root embedding
        l_root = inputs[3]
        # l_merge
        l_merge = inputs[4]

        # num_steps
        num_steps = l_in.shape[1]

        """alphas_e1"""
        # e1_a=e*w:(batch_size*num_steps,attention_size2)
        l_e1_a, updates = theano.scan(
            lambda i, x, y, z: T.dot(T.tanh(T.dot(T.dot(x[i], T.reshape(y[i], (-1, 1))), z)), self.e1_ws2), \
            outputs_info=None, \
            sequences=T.arange(l_in.shape[0]), \
            non_sequences=[l_in, l_entity_e1, self.u_e1])

        # alphas_e1=softmax(a):(batch_size,attention_size2,num_steps)
        exps_e1 = T.reshape(T.exp(l_e1_a), (-1, num_steps, self.attention_size2))
        exps_e1, updates = theano.scan(lambda i, x: T.transpose(x[i]), \
                                       outputs_info=None, \
                                       sequences=T.arange(exps_e1.shape[0]), \
                                       non_sequences=[exps_e1])
        alphas_e1 = exps_e1 / T.reshape(T.sum(exps_e1, axis=2), (-1, self.attention_size2, 1))

        """alphas_e2"""
        # e1_a=e*w:(batch_size*num_steps,attention_size2)
        l_e2_a, updates = theano.scan(
            lambda i, x, y, z: T.dot(T.tanh(T.dot(T.dot(x[i], T.reshape(y[i], (-1, 1))), z)), self.e2_ws2), \
            outputs_info=None, \
            sequences=T.arange(l_in.shape[0]), \
            non_sequences=[l_in, l_entity_e2, self.u_e2])

        # alphas_e1=softmax(a):(batch_size,attention_size2,num_steps)
        exps_e2 = T.reshape(T.exp(l_e2_a), (-1, num_steps, self.attention_size2))
        exps_e2, updates = theano.scan(lambda i, x: T.transpose(x[i]), \
                                       outputs_info=None, \
                                       sequences=T.arange(exps_e2.shape[0]), \
                                       non_sequences=[exps_e2])
        alphas_e2 = exps_e2 / T.reshape(T.sum(exps_e2, axis=2), (-1, self.attention_size2, 1))

        """alphas_root"""
        # e1_a=e*w:(batch_size*num_steps,attention_size2)
        l_root_a, updates = theano.scan(
            lambda i, x, y, z: T.dot(T.tanh(T.dot(T.dot(x[i], T.reshape(y[i], (-1, 1))), z)), self.root_ws2), \
            outputs_info=None, \
            sequences=T.arange(l_in.shape[0]), \
            non_sequences=[l_in, l_root, self.u_root])

        # alphas_root=softmax(a):(batch_size,attention_size2,num_steps)
        exps_root = T.reshape(T.exp(l_root_a), (-1, num_steps, self.attention_size2))
        exps_root, updates = theano.scan(lambda i, x: T.transpose(x[i]), \
                                         outputs_info=None, \
                                         sequences=T.arange(exps_root.shape[0]), \
                                         non_sequences=[exps_root])
        alphas_root = exps_root / T.reshape(T.sum(exps_root, axis=2), (-1, self.attention_size2, 1))

        # r_e1:(batch_size,attention_size2,gru_size)
        r_e1, updates = theano.scan(lambda i, x, y: T.dot(y[i], x[i]), \
                                    outputs_info=None, \
                                    sequences=T.arange(l_in.shape[0]), \
                                    non_sequences=[l_merge, alphas_e1])
        r_e2, updates = theano.scan(lambda i, x, y: T.dot(y[i], x[i]), \
                                    outputs_info=None, \
                                    sequences=T.arange(l_in.shape[0]), \
                                    non_sequences=[l_merge, alphas_e2])
        r_root, updates = theano.scan(lambda i, x, y: T.dot(y[i], x[i]), \
                                      outputs_info=None, \
                                      sequences=T.arange(l_in.shape[0]), \
                                      non_sequences=[l_merge, alphas_root])
        # r:(batch_szie,attention_size2,3*gru_size)
        r = T.concatenate((r_e1, r_root, r_e2,), axis=2)

        return r

    def get_output_shape_for(self, input_shapes):
        shapes_fir = input_shapes[4]
        return (shapes_fir[0], self.attention_size2, 3 * shapes_fir[2])


class SelfAttEntRootLayer3(lasagne.layers.MergeLayer):

    def get_output_for(self, inputs, **kwargs):
        l_att_matrix = inputs[0]
        l_merge = inputs[1]

        r, updates = theano.scan(lambda i, x, y: T.dot(y[i], x[i]), \
                                 outputs_info=None, \
                                 sequences=T.arange(l_merge.shape[0]), \
                                 non_sequences=[l_merge, l_att_matrix])

        return r

    def get_output_shape_for(self, input_shapes):
        shapes_fir = input_shapes[1]
        shapes_sec = input_shapes[0]
        return (shapes_fir[0], shapes_sec[1], shapes_fir[2])


class SelfAttEntRootLayer2(lasagne.layers.MergeLayer):

    def __init__(self, incomings, Ws1=lasagne.init.Normal(), Ws2=lasagne.init.Normal(), attention_size2=20, **kwargs):
        super(SelfAttEntRootLayer2, self).__init__(incomings, **kwargs)
        self.attention_size1 = 350
        self.attention_size2 = attention_size2
        self.Ws1 = self.add_param(Ws1, (self.attention_size1, 3), name='Ws1')
        self.Ws2 = self.add_param(Ws2, (self.attention_size2, self.attention_size1), name='Ws2')

    def get_output_for(self, inputs, **kwargs):
        # input
        l_in = inputs[0]
        # entity e1 embedding
        l_entity_e1 = inputs[1]
        # entity e2 embedding
        l_entity_e2 = inputs[2]
        # root embedding
        l_root = inputs[3]
        # l_merge
        l_merge = inputs[4]

        # num_steps
        num_steps = l_in.shape[1]

        """f=e*w"""
        # e1_a=e*w:(batch_size,num_steps,1)
        l_e1_a, updates = theano.scan(lambda i, x, y: T.dot(x[i], T.reshape(y[i], (-1, 1))), \
                                      outputs_info=None, \
                                      sequences=T.arange(l_in.shape[0]), \
                                      non_sequences=[l_in, l_entity_e1])

        # l_e2_a=e*w:(batch_size,num_steps,1)
        l_e2_a, updates = theano.scan(lambda i, x, y: T.dot(x[i], T.reshape(y[i], (-1, 1))), \
                                      outputs_info=None, \
                                      sequences=T.arange(l_in.shape[0]), \
                                      non_sequences=[l_in, l_entity_e2])

        # l_root_a=e*w:(batch_size,num_steps,1)
        l_root_a, updates = theano.scan(lambda i, x, y: T.dot(x[i], T.reshape(y[i], (-1, 1))), \
                                        outputs_info=None, \
                                        sequences=T.arange(l_in.shape[0]), \
                                        non_sequences=[l_in, l_root])

        """F=concatenate"""
        # F:(batch_size,num_steps,3)
        F = T.concatenate((l_e1_a, l_e2_a, l_root_a), axis=2)
        #        F=(l_e1_a+l_e2_a+l_root_a)/3.0

        """S"""
        # S:(batch_size,attention2,num_steps)
        S, updates = theano.scan(lambda i, x: T.dot(self.Ws2, T.tanh(T.dot(self.Ws1, T.transpose(x[i])))), \
                                 outputs_info=None, \
                                 sequences=T.arange(F.shape[0]), \
                                 non_sequences=[F])

        """A"""
        S = T.reshape(S, (-1, num_steps))
        # A:(batch_size,attention2,num_steps)
        A = T.reshape(T.nnet.softmax(S), (-1, self.attention_size2, num_steps))

        # r=l_in*alpha:(batch_size,attention_size2,gru_size)
        r, updates = theano.scan(lambda i, x, y: T.dot(y[i], x[i]), \
                                 outputs_info=None, \
                                 sequences=T.arange(l_merge.shape[0]), \
                                 non_sequences=[l_merge, A])

        return r

    def get_output_shape_for(self, input_shapes):
        shapes_fir = input_shapes[4]
        return (shapes_fir[0], self.attention_size2, shapes_fir[2])


class FrobeniusLayer(lasagne.layers.Layer):
    """
    FrobeniusLayer
    """

    def __init__(self, incoming, attention_size2=20, **kwargs):
        super(FrobeniusLayer, self).__init__(incoming, **kwargs)
        self.attention_size2 = attention_size2

    def get_output_for(self, input, **kwargs):
        """
        Frobenius
        """
        l_alpha = input
        l_alphaT = T.transpose(l_alpha, (0, 2, 1))

        #        #Eye matrix:(batch_size,col,col)
        #        eye_matrix,updates=theano.scan(lambda i:T.eye(self.attention_size2),\
        #           outputs_info=None,\
        #           sequences=T.arange(l_alpha.shape[0]),\
        #           non_sequences=None)

        # A*A^T
        l_alpha, updates = theano.scan(lambda i, x, y: T.dot(x[i], y[i]), \
                                       outputs_info=None, \
                                       sequences=T.arange(l_alpha.shape[0]), \
                                       non_sequences=[l_alpha, l_alphaT])

        mat, updates = theano.scan(lambda i, x: x[i] - T.eye(self.attention_size2), \
                                   outputs_info=None, \
                                   sequences=T.arange(l_alpha.shape[0]), \
                                   non_sequences=[l_alpha])

        size = mat.shape
        ret = (T.sum(T.sum(mat ** 2, axis=2), axis=1).squeeze() + 1e-10) ** 0.5
        return T.mean(T.sum(ret))

    #        return l_alpha

    def get_output_shape_for(self, input_shape):
        return (1)
