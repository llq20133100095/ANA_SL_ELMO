#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:07:50 2018

@author: llq
"""
import theano.tensor as T
import theano

def Negative_loss(coding_dist,true_dist,alpha,lamda):
    """
    Negative loss:
        -sum[alpha * ture_dist * sigmoid(-y_pre)^lamda * log(y_pre)]
    
    Parameters:
        alpha:control the negative number.(1,1,1,1....,a)
    """ 
    constant_alpha=3
    k=1
    if true_dist.ndim == coding_dist.ndim:
        loss=T.sum(alpha * constant_alpha * true_dist * T.pow((T.nnet.sigmoid(-k * coding_dist)),lamda) * T.log(coding_dist),axis=coding_dist.ndim - 1)
        
        return loss
        
def Margin_loss(coding_dist,true_dist):
    
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
        '''
        #Calculation: predictioin to true_label
        y_pre2true=T.sum(true_dist * coding_dist, axis=1)
        
        #Calculation: prediction to false_label
        y_pre2false=T.max((1-true_dist) * coding_dist, axis=1)
        
        loss=1+y_pre2true-y_pre2false
        '''        
        #Calculation: predictioin to true_label
#        y_pre2true=T.sum(true_dist * T.log(1+T.exp(2*(3-coding_dist))),axis=1)        
        true_pre=T.sum(true_dist * coding_dist,axis=1)
        y_pre2true=T.sum(true_dist * T.exp((3-coding_dist)),axis=1)

#        #Negative loss in y_pre2true
#        y_pre2true=T.nnet.sigmoid(y_pre2true)*y_pre2true

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
#        y_pre2false=T.log(1+T.exp(2*(0.5+coding_dist_true2inf)))
        y_pre2false=T.exp((0.25+coding_dist_true2inf))
                
        #Negative loss in y_pre2false
#        y_pre2false=T.nnet.sigmoid(k*y_pre2false)*y_pre2false
        
        #SSL
        stimulative=T.exp(coding_dist_true2inf-true_pre)
#        loss=4*T.nnet.sigmoid(y_pre2true)*T.nnet.sigmoid(y_pre2false)*stimulative*T.log(1+y_pre2true+y_pre2false)

        loss=0.5*T.nnet.sigmoid(y_pre2true)*T.nnet.sigmoid(y_pre2false)*T.log(1+y_pre2true+y_pre2false)
        
        return loss,coding_dist_true2inf,true_pre
    
    else:
        print "true_dist.ndim != coding_dist.ndim"

def Stimulation_loss(coding_dist,true_dist):
    
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
        """"""
        coding_dist1=T.tanh(coding_dist)
        y_pre2true=T.sum(true_dist * T.exp((-coding_dist1)),axis=1)
        
        #search the true label id
        true_label_id=T.argmax(true_dist,axis=1)
        #persist the false label in coding_dist
        coding_dist_false=(1-true_dist) * coding_dist1
        #set true label to "-inf"
        coding_dist_true2inf,updates=theano.scan(set_inf_in1dim,\
           outputs_info=None,\
           sequences=T.arange(coding_dist_false.shape[0]),\
           non_sequences=[coding_dist_false,true_label_id])
        #search the max in false label
        coding_dist_true2inf=T.max(coding_dist_true2inf,axis=1)
        y_pre2false=T.exp((coding_dist_true2inf))
        
        
        """stimulative"""
        coding_dist=T.nnet.softmax(coding_dist)
        
        #Calculation: predictioin to true_label
        true_pre=T.sum(true_dist * coding_dist,axis=1)
#        y_pre2true=T.sum(true_dist * T.exp((3-coding_dist)),axis=1)

        #search the true label id
        true_label_id=T.argmax(true_dist,axis=1)
        #persist the false label in coding_dist
        coding_dist_false=(1-true_dist) * coding_dist
        #set true label to "-inf"
        coding_dist_true2inf,updates=theano.scan(set_inf_in1dim,\
           outputs_info=None,\
           sequences=T.arange(coding_dist_false.shape[0]),\
           non_sequences=[coding_dist_false,true_label_id])
        #search the max in false label
        coding_dist_true2inf=T.max(coding_dist_true2inf,axis=1)
#        y_pre2false=T.exp((0.25+coding_dist_true2inf))
        
        #SSL
        stimulative=1+coding_dist_true2inf-true_pre
        
#        loss=stimulative*(-T.log(1e-8+true_pre))
        loss=stimulative*T.log(1+y_pre2true+y_pre2false)
        
        return loss,y_pre2false,y_pre2true,stimulative
    
    else:
        print "true_dist.ndim != coding_dist.ndim"
        
def kl_loss_compute(logits1, logits2, true_dist):
    """ 
    KL loss
    """
    pred1=T.nnet.softmax(logits1)
    pred2=T.nnet.softmax(logits2)
    loss = T.mean(T.sum(pred2 * T.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))

#    pred1 = T.nnet.softmax(logits1)
#    pred2 = T.nnet.softmax(logits2)
#    pred1=T.sum(true_dist * pred1,axis=1)
#    pred2=T.sum(true_dist * pred2,axis=1)
#    loss = T.mean(pred2 * T.log(1 + pred2 / (pred1 + 1e-8)))
    
#    pred1=logits1
#    pred2=logits2
#    loss = T.mean(T.sum(pred2 * T.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))
    return loss
        
def SSL_Mutual(coding_dist,true_dist):
    
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
        '''
        #Calculation: predictioin to true_label
        y_pre2true=T.sum(true_dist * coding_dist, axis=1)
        
        #Calculation: prediction to false_label
        y_pre2false=T.max((1-true_dist) * coding_dist, axis=1)
        
        loss=1+y_pre2true-y_pre2false
        '''        
        #Calculation: predictioin to true_label
#        y_pre2true=T.sum(true_dist * T.log(1+T.exp(2*(3-coding_dist))),axis=1)        
        true_pre=T.sum(true_dist * coding_dist,axis=1)
        y_pre2true=T.sum(true_dist * T.exp((3-coding_dist)),axis=1)

#        #Negative loss in y_pre2true
#        y_pre2true=T.nnet.sigmoid(y_pre2true)*y_pre2true

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
#        y_pre2false=T.log(1+T.exp(2*(0.5+coding_dist_true2inf)))
        y_pre2false=T.exp((0.25+coding_dist_true2inf))

        #Negative loss in y_pre2false
#        y_pre2false=T.nnet.sigmoid(k*y_pre2false)*y_pre2false
        
        #SSL
        stimulative=T.exp(2+coding_dist_true2inf-true_pre)
#        loss=4*T.nnet.sigmoid(y_pre2true)*T.nnet.sigmoid(y_pre2false)*stimulative*T.log(1+y_pre2true+y_pre2false)
        loss=0.5*T.nnet.sigmoid(y_pre2true)*T.nnet.sigmoid(y_pre2false)*T.log(1+y_pre2true+y_pre2false)
       
        return loss,coding_dist,true_pre
    
    else:
        print "true_dist.ndim != coding_dist.ndim"
  
def kl_loss_compute2(logits1, logits2, logits3):
    """ 
    KL loss
    """
#    pred1=T.nnet.softmax(logits1)
#    pred2=T.nnet.softmax(logits2)
#    pred3=T.nnet.softmax(logits3)
#    loss1 = T.mean(T.sum(pred2 * T.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))
#    loss2 = T.mean(T.sum(pred3 * T.log(1e-8 + pred3 / (pred1 + 1e-8)), 1))
#    return (loss1+loss2)/2.0

    pred1=T.nnet.softmax(logits1)
    pred2=T.nnet.softmax(logits2)
    loss = T.mean(T.sum(pred2 * T.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))
    return loss
    
def SSL_Mutual2(coding_dist,true_dist):
    
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
        """"""
        coding_dist1=T.tanh(coding_dist)
        y_pre2true=T.sum(true_dist * T.exp((-coding_dist1)),axis=1)
        
        #search the true label id
        true_label_id=T.argmax(true_dist,axis=1)
        #persist the false label in coding_dist
        coding_dist_false=(1-true_dist) * coding_dist1
        #set true label to "-inf"
        coding_dist_true2inf,updates=theano.scan(set_inf_in1dim,\
           outputs_info=None,\
           sequences=T.arange(coding_dist_false.shape[0]),\
           non_sequences=[coding_dist_false,true_label_id])
        #search the max in false label
        coding_dist_true2inf=T.max(coding_dist_true2inf,axis=1)
        y_pre2false=T.exp((coding_dist_true2inf))
        
        
        """stimulative"""
        coding_dist=T.nnet.softmax(coding_dist)
        
        #Calculation: predictioin to true_label
        true_pre=T.sum(true_dist * coding_dist,axis=1)
#        y_pre2true=T.sum(true_dist * T.exp((3-coding_dist)),axis=1)

        #search the true label id
        true_label_id=T.argmax(true_dist,axis=1)
        #persist the false label in coding_dist
        coding_dist_false=(1-true_dist) * coding_dist
        #set true label to "-inf"
        coding_dist_true2inf,updates=theano.scan(set_inf_in1dim,\
           outputs_info=None,\
           sequences=T.arange(coding_dist_false.shape[0]),\
           non_sequences=[coding_dist_false,true_label_id])
        #search the max in false label
        coding_dist_true2inf=T.max(coding_dist_true2inf,axis=1)
#        y_pre2false=T.exp((0.25+coding_dist_true2inf))
        
        #SSL
        stimulative=1+coding_dist_true2inf-true_pre
        
#        loss=stimulative*(-T.log(1e-8+true_pre))
        loss=stimulative*T.log(1+y_pre2true+y_pre2false)
        
        return loss,y_pre2false,y_pre2true,stimulative
    
    else:
        print "true_dist.ndim != coding_dist.ndim"
