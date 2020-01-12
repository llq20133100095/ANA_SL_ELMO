#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 09:49:05 2018

1.a pairwise ranking loss function
2.a novel distance function
3.Focal loss

@author: llq
"""
import theano.tensor as T
import theano
import lasagne
from collections import OrderedDict
import numpy as np

def focal_loss(coding_dist,true_dist):
    
    if true_dist.ndim == coding_dist.ndim:
        
        ce_loss=T.sum(true_dist * (coding_dist+1e-14),axis=1)
        gamma = 2
        alpha = 0.25
        loss=-alpha * T.power(1-ce_loss,gamma) *  T.log(ce_loss)
        
        return loss,ce_loss,loss,-T.log(ce_loss)
    
    else:
        print "true_dist.ndim != coding_dist.ndim"

        
def ranking_loss(coding_dist,true_dist):
    
    def set_inf_in2dim(j,coding_dist,true_label_id):
        """
        Search true_label_id==j,and set coding_dist[i][j]="-inf" 
        """
        return T.switch(T.eq(j,true_label_id),T.constant(float("-inf")),coding_dist[j])
        
    def set_inf_in1dim(i,coding_dist,true_label_id):
        #coding_dist[:,label_id] doesn't become "-0.0"
        loss_margin,updates=theano.scan(set_inf_in2dim,\
           outputs_info=None,\
           sequences=T.arange(coding_dist.shape[1]),\
           non_sequences=[coding_dist[i],true_label_id[i]])
        return loss_margin
        
    if true_dist.ndim == coding_dist.ndim:

        #Calculation: predictioin to true_label
        true_pre=T.sum(true_dist * coding_dist,axis=1)
        y_pre2true=T.sum(true_dist * T.exp((3-coding_dist)),axis=1)

        #search the true label id
        true_label_id=T.argmax(true_dist,axis=1)
        #persist the false label in coding_dist
        coding_dist=(1-true_dist) * coding_dist
        #set true label to "-inf"
        coding_dist_true2inf,updates=theano.scan(set_inf_in1dim,\
           outputs_info=None,\
           sequences=T.arange(coding_dist.shape[0]),\
           non_sequences=[coding_dist,true_label_id])
        #search the max in false label
        coding_dist_true2inf=T.max(coding_dist_true2inf,axis=1)
        #Calculation: predictioin to false_label
        y_pre2false=T.exp((0.25+coding_dist_true2inf))
                

        loss=T.log(1+y_pre2true+y_pre2false)
        
        return loss,coding_dist_true2inf,true_pre,loss

def distance_loss(coding_dist,true_dist):
    
    def set_inf_in2dim(j,coding_dist,true_label_id):
        """
        Search true_label_id==j,and set coding_dist[i][j]="-inf" 
        """
        return T.switch(T.eq(j,true_label_id),T.constant(float("-inf")),coding_dist[j])
        
    def set_inf_in1dim(i,coding_dist,true_label_id):
        #coding_dist[:,label_id] doesn't become "-0.0"
        loss_margin,updates=theano.scan(set_inf_in2dim,\
           outputs_info=None,\
           sequences=T.arange(coding_dist.shape[1]),\
           non_sequences=[coding_dist[i],true_label_id[i]])
        return loss_margin
    
    def compare_max(l2_norm,coding_dist):
        
        result,updates=theano.scan(lambda i,x:T.switch(T.le(x[i],T.constant(1e-12)),T.constant(1e-12),x[i]),\
           outputs_info=None,\
           sequences=T.arange(coding_dist.shape[0]),\
           non_sequences=[l2_norm])
        return result
        
    if true_dist.ndim == coding_dist.ndim:
        
        #L2-norm
        l2_norm=T.sqrt(T.sum(T.power(coding_dist,2),axis=1))
        l2_norm=compare_max(l2_norm,coding_dist)
        #label-norm
        
        #Calculation: predictioin to true_label
        true_pre=T.sum(true_dist * coding_dist,axis=1)
        y_pre2true=T.sqrt(T.power((true_pre/l2_norm)-1,2))

        #search the true label id
        true_label_id=T.argmax(true_dist,axis=1)
        #persist the false label in coding_dist
        coding_dist=coding_dist/T.reshape(l2_norm,(100,1))
        coding_dist=(1-true_dist) * coding_dist
        #set true label to "-inf"
        coding_dist_true2inf,updates=theano.scan(set_inf_in1dim,\
           outputs_info=None,\
           sequences=T.arange(coding_dist.shape[0]),\
           non_sequences=[coding_dist,true_label_id])
        #search the max in false label
        coding_dist_true2inf=T.max(coding_dist_true2inf,axis=1)
        #Calculation: predictioin to false_label
        y_pre2false=T.sqrt(T.power(coding_dist_true2inf-1,2))
                

        loss=1+y_pre2true-y_pre2false
        
        return loss,coding_dist_true2inf,true_pre,loss

def max_f1_score_loss(coding_dist, true_dist):
    """
    f1 = (2 * N_WWW) / (N_D + N_W)
    """
    def set_2dim(j, matrix):
        return T.switch(T.eq(matrix[j], 0), -100, T.log(matrix[j]))
        
    def set_1dim(i,matrix):
        matrix_2dim, updates=theano.scan(set_2dim,\
           outputs_info=None,\
           sequences=T.arange(matrix.shape[1]),\
           non_sequences=[matrix[i]])
        return matrix_2dim
    
    if true_dist.ndim == coding_dist.ndim:
        '''
        #parameter
        tau=-2
        #ce_loss: [batch_size,]
        ce_loss = T.log(T.sum(true_dist * (coding_dist + 1e-14), axis=1))
        
        #N_D : >tau=1; <tau=0
        N_d = ce_loss
        N_D, updates = theano.scan(lambda i, x, y: T.switch(T.lt(x[i], y), 0, 1),\
           outputs_info=None,\
           sequences=T.arange(N_d.shape[0]),\
           non_sequences=[N_d,tau])
        N_D = T.sum(N_D, axis=0)
#        N_D = T.nnet.sigmoid(N_D)
        
        #N_W: [1, class_numbers]
        N_W = T.sum(true_dist ,axis=0)
        
        #N_WWW: [1, class_numbers]
        matrix = true_dist * (coding_dist + 1e-14)
        matrix_log, updates = theano.scan(set_1dim,\
           outputs_info=None,\
           sequences=T.arange(matrix.shape[0]),\
           non_sequences=[matrix])
        #indictor function
        I = T.switch(T.lt(matrix_log, tau), 0, 1)
        N_WWW = T.sum(I, axis=0)
#        N_WWW = T.nnet.sigmoid(N_WWW)
        '''

        classid = T.argmax(coding_dist, axis=1)
        
        #N_D           
        def N_D_count_2dim(j, x, y):
            return T.switch(T.eq(j, y), 1, 0)
        
        def N_D_count_1dim(i, x, y):
            N_d, updates = theano.scan(N_D_count_2dim,\
               outputs_info=None,\
               sequences=T.arange(x.shape[1]),\
               non_sequences=[x[i], y[i]])
            return N_d
        
        predict, updates = theano.scan(N_D_count_1dim,\
           outputs_info=None,\
           sequences=T.arange(coding_dist.shape[0]),\
           non_sequences=[coding_dist, classid])
        N_D = T.reshape(T.sum(predict, axis=0), (1, -1))
        N_D = T.nnet.sigmoid(N_D)
        
        #N_W: [1, class_numbers]
        N_W = T.reshape(T.sum(true_dist ,axis=0), (1, -1))
        
        #N_WWW: [1, class_numbers]
        matrix = true_dist * predict
        N_WWW = T.reshape(T.sum(matrix, axis=0), (1,-1))
        N_WWW = T.nnet.sigmoid(N_WWW)
        
        #F1-SCORE
        F1 = T.true_div(2 * N_WWW, (N_D + N_W + 1e-14))
        
        return F1, F1
    
    else:
        print "true_dist.ndim != coding_dist.ndim"

def sga(loss_or_grads, params, learning_rate):
    """Stochastic Gradient Descent (SGD) updates
    """
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param + learning_rate * grad

    return updates