2#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:07:41 2018

@author: llq
"""

import lasagne
import theano.tensor as T
import theano
        
class SplitInLeft(lasagne.layers.MergeLayer):
    """
    Get the left SDP.
    And the rest is zeros.
    """ 
    def get_output_for(self,inputs,**kwargs):

        #the left sdp length
        input_left_length=inputs[0]
        l_gru=inputs[1]

        #batch_size
        batch_size=T.arange(l_gru.shape[0])
        
        def left_pass(n,left_length,gru):
            #Retain the left sdp
            pass_one=T.ones((left_length[n][0],gru.shape[2]))
            #delete the rest
            no_pass_zero=T.zeros((gru.shape[1]-left_length[n][0],gru.shape[2]))

            mask=T.concatenate((pass_one,no_pass_zero),axis=0)
            return gru[n]*mask
                    
        #split the left sdp
        output_left_sdp,updates_left=theano.scan(left_pass,\
           outputs_info=None,\
           sequences=batch_size,\
           non_sequences=[input_left_length,l_gru])

            
        return output_left_sdp
        
    def get_output_shape_for(self, input_shapes):
        l_gru=input_shapes[1]
#        shapes_sec=input_shapes[1]
        
        output_shape=(l_gru[0],l_gru[1],l_gru[2])
        return output_shape
    
class SplitInRight(lasagne.layers.MergeLayer):
    """
    Get the right SDP.
    And the rest is zeros.
    """ 
    
    def get_output_for(self,inputs,**kwargs):

        #the left sdp length
        input_left_length=inputs[0]
        input_sen_length=inputs[1]
        l_gru=inputs[2]

        #batch_size
        batch_size=T.arange(l_gru.shape[0])
        
        def right_pass(n,left_length,sen_length,gru):
            #delete the left sdp(because has root,it must sub 1)
            no_pass_left=T.zeros((left_length[n][0]-1,gru.shape[2]))
            #Retain the right sdp
            pass_right=T.ones((sen_length[n][0]-left_length[n][0]+1,gru.shape[2]))
            #delete the rest
            rest=T.zeros((gru.shape[1]-sen_length[n][0],gru.shape[2]))
            
            mask=T.concatenate((no_pass_left,pass_right,rest),axis=0)
            
            return gru[n]*mask
       
        #split the rest sdp
        output_right_sdp,updates_right=theano.scan(right_pass,\
           outputs_info=None,\
           sequences=batch_size,\
           non_sequences=[input_left_length,input_sen_length,l_gru])
        
        return output_right_sdp
        
    def get_output_shape_for(self, input_shapes):
        l_gru=input_shapes[2]
#        shapes_sec=input_shapes[1]
        
        output_shape=(l_gru[0],l_gru[1],l_gru[2])
        return output_shape
 

class SplitInGlobal(lasagne.layers.MergeLayer):
    """
    Get the global SDP.
    And the rest is zeros.
    """ 
    
    def get_output_for(self,inputs,**kwargs):

        #the sdp length
        input_sen_length=inputs[0]
        l_gru=inputs[1]

        #batch_size
        batch_size=T.arange(l_gru.shape[0])
        
        def global_pass(n,sen_length,gru):
            #Retain the global sdp
            pass_right=T.ones((sen_length[n][0],gru.shape[2]))
            #delete the rest
            rest=T.zeros((gru.shape[1]-sen_length[n][0],gru.shape[2]))
            
            mask=T.concatenate((pass_right,rest),axis=0)
            
            return gru[n]*mask
       
        #split the rest sdp
        output_global_sdp,updates_right=theano.scan(global_pass,\
           outputs_info=None,\
           sequences=batch_size,\
           non_sequences=[input_sen_length,l_gru])
        
        return output_global_sdp
        
    def get_output_shape_for(self, input_shapes):
        l_gru=input_shapes[1]
        
        output_shape=(l_gru[0],l_gru[1],l_gru[2])
        return output_shape
        
class HighwayNetwork1D(lasagne.layers.Layer):
    """
    Highway network
    1.z=t*H(x)+(1-t)*x
    2.H(x)=tanh(W*x+b)
    3.t=sigmoid(W*x+b)
    """
    def __init__(self, incoming, h_w=lasagne.init.Normal(), h_b=lasagne.init.Normal(), t_w=lasagne.init.Normal(), t_b=lasagne.init.Normal(), **kwargs):
        super(HighwayNetwork1D,self).__init__(incoming, **kwargs)
#        num_steps = self.input_shape[1]
        cnn_gru_size = self.input_shape[1]
        self.h_w = self.add_param(h_w, (cnn_gru_size,), name='h_w')
        self.h_b=self.add_param(h_b, (cnn_gru_size,), name='h_b')
        self.t_w=self.add_param(t_w, (cnn_gru_size,), name='t_w')
        self.t_b=self.add_param(t_b, (cnn_gru_size,), name='t_b')
        
    def get_output_for(self, input, **kwargs):
#        #batch_size
#        batch_size=T.arange(input.shape[0])
        
        #H(x)=tanh(W*x+b)
        h_x=T.tanh(self.h_w*input+self.h_b)
        #t=sigmoid(W*x+b)
        t=T.nnet.sigmoid(self.t_w*input+self.t_b)
        #z=t*H(x)+(1-t)*x
        z=t*h_x+(1-t)*input

        return z
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])
        
class HighwayNetwork2D(lasagne.layers.Layer):
    """
    Highway network  and use it has 3D
    1.z=t*H(x)+(1-t)*x
    2.H(x)=tanh(W*x+b)
    3.t=sigmoid(W*x+b)
    """
    def __init__(self, incoming, h_w=lasagne.init.Normal(), h_b=lasagne.init.Normal(), t_w=lasagne.init.Normal(), t_b=lasagne.init.Normal(), **kwargs):
        super(HighwayNetwork2D,self).__init__(incoming, **kwargs)
        num_filters = self.input_shape[1]
        cnn_size = self.input_shape[2]
        self.h_w = self.add_param(h_w, (num_filters,cnn_size), name='h_w')
        self.h_b=self.add_param(h_b, (cnn_size,), name='h_b')
        self.t_w=self.add_param(t_w, (num_filters,cnn_size), name='t_w')
        self.t_b=self.add_param(t_b, (cnn_size,), name='t_b')
        
    def get_output_for(self, input, **kwargs):
#        #batch_size
#        batch_size=T.arange(input.shape[0])
        
        #H(x)=tanh(W*x+b)
        h_x=T.tanh(self.h_w*input+self.h_b)
        #t=sigmoid(W*x+b)
        t=T.nnet.sigmoid(self.t_w*input+self.t_b)
        #z=t*H(x)+(1-t)*x
        z=t*h_x+(1-t)*input

        return z
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])
        
class DeleteFirstInCNN(lasagne.layers.Layer):
    """
    Delete the first line in ARC-ONE CNN
    """
    def __init__(self, incoming, size, **kwargs):
        super(DeleteFirstInCNN,self).__init__(incoming, **kwargs)
        self.size=size
        
    def get_output_for(self, input, **kwargs):
        #batch_size
        batch_size=T.arange(input.shape[0])
        
        #split the rest sdp
        output,updates_right=theano.scan(lambda i,x:x[i][0][1:],\
           outputs_info=None,\
           sequences=batch_size,\
           non_sequences=[input])
        
        return output
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.size, input_shape[3])

        
class MarginLossLayer(lasagne.layers.Layer):
    """
    w_classes is a parameter to be learned by the network.
        In paper <<Classifying Relations by Ranking with Convolutional Neural Networks>>.
        Set the new loss layer.
    """
    def __init__(self, incoming, w_classes=lasagne.init.Uniform(), class_number=19, **kwargs):
        super(MarginLossLayer,self).__init__(incoming, **kwargs)
        #the network output
        network_output_shape = self.input_shape
        
        self.class_number=class_number
        
#        #init w_classes: it's format is (class_number,network_output[1])
#        w_classes=lasagne.init.Uniform(T.sqrt(6/(class_number+network_output[1])))
        self.w_classes=self.add_param(w_classes, (class_number,network_output_shape[1]), name='w_classes')
        
    
    def get_output_for(self, input, **kwargs):
        #the network output  (batch_size,None)
        network_output = input
        y_pre2each_label=T.dot(network_output,T.transpose(self.w_classes))
        '''
        """w_o/|w_o|"""
        #the norm of network_output[i]
        network_output_norm=T.sqrt(T.sum(T.pow(network_output,2),axis=1))
        #normalize the output
#        network_max=T.max(network_output,axis=1)
#        network_min=T.min(network_output,axis=1)
#        network_output,updates=theano.scan(lambda i,x,y,z:
#           (x[i]-z[i])/(y[i]-z[i]),\
#           outputs_info=None,\
#           sequences=T.arange(network_output.shape[0]),\
#           non_sequences=[network_output,network_max,network_min])
        network_output,updates=theano.scan(lambda i,x,y:
           x[i]/y[i],\
           outputs_info=None,\
           sequences=T.arange(network_output.shape[0]),\
           non_sequences=[network_output,network_output_norm])
              
        
        """normaliza the w_classes to 0~1"""
        w_classes=self.w_classes
#        w_max=T.max(w_classes,axis=1)
#        w_min=T.min(w_classes,axis=1)
#        w_classes_norm,updates=theano.scan(lambda i,x,y,z:
#           (x[i]-z[i])/(y[i]-z[i]),\
#           outputs_info=None,\
#           sequences=T.arange(w_classes.shape[0]),\
#           non_sequences=[w_classes,w_max,w_min])
        
        """calculate the margin between prediction and each label: 
                y_pre2each_label=||w_o/|w_o|-w_classes_norm||
        """
#        #get the embedding of "true label" in w_classes
#        true_label_embedding=T.dot(target_1hot,w_classes_norm)
        
        batch_size=T.arange(network_output.shape[0])
        
        #Calculate the margin in each batch
        y_pre2each_label,updates=theano.scan(lambda i,x,y:\
           T.sqrt(T.sum(T.pow((y-x[i]),2),axis=1)),\
           outputs_info=None,\
           sequences=batch_size,\
           non_sequences=[network_output,w_classes])
        
#        #normalizate the y_pre2each_label to 0~1
#        y_pre2each_label_max=T.max(y_pre2each_label,axis=1)
#        y_pre2each_label_min=T.min(y_pre2each_label,axis=1)
#        y_pre2each_label,updates=theano.scan(lambda i,x,y,z:
#           (x[i]-z[i])/(y[i]-z[i]),\
#           outputs_info=None,\
#           sequences=T.arange(y_pre2each_label.shape[0]),\
#           non_sequences=[y_pre2each_label,y_pre2each_label_max,y_pre2each_label_min])
        '''
        return y_pre2each_label
        
    def get_output_shape_for(self, input_shape):
        output_shape=(input_shape[0],self.class_number)
#        output_shape=(19,300)
        return output_shape