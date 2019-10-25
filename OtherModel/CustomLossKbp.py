#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 22:06:16 2018

@author: llq
"""
import theano.tensor as T
import theano


def kl_loss_compute(logits1, logits2):
    """
    KL loss
    """
    pred1 = T.nnet.softmax(logits1)
    pred2 = T.nnet.softmax(logits2)
    loss = T.mean(T.sum(pred2 * T.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))

    return loss


def SSL(coding_dist, true_dist, epoch):
    def set_inf_in2dim(j, coding_dist, true_label_id):
        """
        Search true_label_id==j,and set coding_dist[i][j]="-inf"
        """
        return T.switch(T.eq(j, true_label_id), T.constant(float("-inf")), coding_dist[j])

    def set_inf_in1dim(i, coding_dist, true_label_id):
        # coding_dist[:,label_id] doesn't become "-0.0"
        loss_margin, updates = theano.scan(set_inf_in2dim, \
                                           outputs_info=None, \
                                           sequences=T.arange(coding_dist.shape[1]), \
                                           non_sequences=[coding_dist[i], true_label_id[i]])
        return loss_margin

    if true_dist.ndim == coding_dist.ndim:
        '''
        #Calculation: predictioin to true_label
        y_pre2true=T.sum(true_dist * coding_dist, axis=1)

        #Calculation: prediction to false_label
        y_pre2false=T.max((1-true_dist) * coding_dist, axis=1)

        loss=1+y_pre2true-y_pre2false
        '''
        # Calculation: predictioin to true_label
        #        y_pre2true=T.sum(true_dist * T.log(1+T.exp(2*(3-coding_dist))),axis=1)
        y_pre2true_softmax = T.sum(true_dist * T.nnet.softmax(coding_dist), axis=1)

        true_pre = T.sum(true_dist * coding_dist, axis=1)
        y_pre2true = T.sum(true_dist * T.exp((3 - coding_dist)), axis=1)

        #        #Negative loss in y_pre2true
        #        y_pre2true=T.nnet.sigmoid(y_pre2true)*y_pre2true

        # search the true label id
        true_label_id = T.argmax(true_dist, axis=1)
        # persist the false label in coding_dist
        coding_dist = (1 - true_dist) * coding_dist
        # set true label to "-inf"
        coding_dist_true2inf, updates = theano.scan(set_inf_in1dim, \
                                                    outputs_info=None, \
                                                    sequences=T.arange(coding_dist.shape[0]), \
                                                    non_sequences=[coding_dist, true_label_id])
        # search the max in false label
        coding_dist_true2inf = T.max(coding_dist_true2inf, axis=1)
        # Calculation: predictioin to false_label
        #        y_pre2false=T.log(1+T.exp(2*(0.5+coding_dist_true2inf)))
        y_pre2false = T.exp((0.5 + coding_dist_true2inf))

        # Negative loss in y_pre2false
        #        y_pre2false=T.nnet.sigmoid(k*y_pre2false)*y_pre2false
        stimulative = T.exp(2 + coding_dist_true2inf - true_pre)
        loss = 4 * T.nnet.sigmoid(y_pre2true) * T.nnet.sigmoid(y_pre2false) * stimulative * T.log(
            1 + y_pre2true + y_pre2false)

        #        loss=T.switch(T.le(epoch,40),-T.log(y_pre2true_softmax),
        #          4*T.nnet.sigmoid(y_pre2true)*T.nnet.sigmoid(y_pre2false)*stimulative*T.log(1+y_pre2true+y_pre2false))

        return loss, stimulative, y_pre2false

    else:
        print "true_dist.ndim != coding_dist.ndim"


def SSL2(coding_dist, true_dist):
    def set_inf_in2dim(j, coding_dist, true_label_id):
        """
        Search true_label_id==j,and set coding_dist[i][j]="-inf"
        """
        return T.switch(T.eq(j, true_label_id), T.constant(float("-inf")), coding_dist[j])

    def set_inf_in1dim(i, coding_dist, true_label_id):
        # coding_dist[:,label_id] doesn't become "-0.0"
        loss_margin, updates = theano.scan(set_inf_in2dim, \
                                           outputs_info=None, \
                                           sequences=T.arange(coding_dist.shape[1]), \
                                           non_sequences=[coding_dist[i], true_label_id[i]])
        return loss_margin

    if true_dist.ndim == coding_dist.ndim:
        '''
        #Calculation: predictioin to true_label
        y_pre2true=T.sum(true_dist * coding_dist, axis=1)

        #Calculation: prediction to false_label
        y_pre2false=T.max((1-true_dist) * coding_dist, axis=1)

        loss=1+y_pre2true-y_pre2false
        '''
        # Calculation: predictioin to true_label
        #        y_pre2true=T.sum(true_dist * T.log(1+T.exp(2*(3-coding_dist))),axis=1)
        y_pre2true_softmax = T.sum(true_dist * T.nnet.softmax(coding_dist), axis=1)

        true_pre = T.sum(true_dist * coding_dist, axis=1)
        y_pre2true = T.sum(true_dist * T.exp((3 - coding_dist)), axis=1)

        #        #Negative loss in y_pre2true
        #        y_pre2true=T.nnet.sigmoid(y_pre2true)*y_pre2true

        # search the true label id
        true_label_id = T.argmax(true_dist, axis=1)
        # persist the false label in coding_dist
        coding_dist = (1 - true_dist) * coding_dist
        # set true label to "-inf"
        coding_dist_true2inf, updates = theano.scan(set_inf_in1dim, \
                                                    outputs_info=None, \
                                                    sequences=T.arange(coding_dist.shape[0]), \
                                                    non_sequences=[coding_dist, true_label_id])
        # search the max in false label
        coding_dist_true2inf = T.max(coding_dist_true2inf, axis=1)
        # Calculation: predictioin to false_label
        #        y_pre2false=T.log(1+T.exp(2*(0.5+coding_dist_true2inf)))
        y_pre2false = T.exp((0.5 + coding_dist_true2inf))

        # Negative loss in y_pre2false
        #        y_pre2false=T.nnet.sigmoid(k*y_pre2false)*y_pre2false
        stimulative = T.exp(2 + coding_dist_true2inf - true_pre)
        loss = 4 * T.nnet.sigmoid(y_pre2true) * T.nnet.sigmoid(y_pre2false) * stimulative * T.log(
            1 + y_pre2true + y_pre2false)

        #        loss=2*T.nnet.sigmoid(y_pre2true)*T.nnet.sigmoid(y_pre2false)*T.log(1+y_pre2true+y_pre2false)

        return loss, stimulative, y_pre2false

    else:
        print "true_dist.ndim != coding_dist.ndim"


def SSL_mutual(coding_dist, true_dist, epoch):
    def set_inf_in2dim(j, coding_dist, true_label_id):
        """
        Search true_label_id==j,and set coding_dist[i][j]="-inf"
        """
        return T.switch(T.eq(j, true_label_id), T.constant(float("-inf")), coding_dist[j])

    def set_inf_in1dim(i, coding_dist, true_label_id):
        # coding_dist[:,label_id] doesn't become "-0.0"
        loss_margin, updates = theano.scan(set_inf_in2dim, \
                                           outputs_info=None, \
                                           sequences=T.arange(coding_dist.shape[1]), \
                                           non_sequences=[coding_dist[i], true_label_id[i]])
        return loss_margin

    if true_dist.ndim == coding_dist.ndim:
        '''
        #Calculation: predictioin to true_label
        y_pre2true=T.sum(true_dist * coding_dist, axis=1)

        #Calculation: prediction to false_label
        y_pre2false=T.max((1-true_dist) * coding_dist, axis=1)

        loss=1+y_pre2true-y_pre2false
        '''
        # Calculation: predictioin to true_label
        #        y_pre2true=T.sum(true_dist * T.log(1+T.exp(2*(3-coding_dist))),axis=1)
        y_pre2true_softmax = T.sum(true_dist * T.nnet.softmax(coding_dist), axis=1)

        true_pre = T.sum(true_dist * coding_dist, axis=1)
        y_pre2true = T.sum(true_dist * T.exp((3 - coding_dist)), axis=1)

        #        #Negative loss in y_pre2true
        #        y_pre2true=T.nnet.sigmoid(y_pre2true)*y_pre2true

        # search the true label id
        true_label_id = T.argmax(true_dist, axis=1)
        # persist the false label in coding_dist
        coding_dist = (1 - true_dist) * coding_dist
        # set true label to "-inf"
        coding_dist_true2inf, updates = theano.scan(set_inf_in1dim, \
                                                    outputs_info=None, \
                                                    sequences=T.arange(coding_dist.shape[0]), \
                                                    non_sequences=[coding_dist, true_label_id])
        # search the max in false label
        coding_dist_true2inf = T.max(coding_dist_true2inf, axis=1)
        # Calculation: predictioin to false_label
        #        y_pre2false=T.log(1+T.exp(2*(0.5+coding_dist_true2inf)))
        y_pre2false = T.exp((0.5 + coding_dist_true2inf))

        # Negative loss in y_pre2false
        #        y_pre2false=T.nnet.sigmoid(k*y_pre2false)*y_pre2false
        stimulative = T.exp(2 + coding_dist_true2inf - true_pre)
        loss = 4 * T.nnet.sigmoid(y_pre2true) * T.nnet.sigmoid(y_pre2false) * stimulative * T.log(
            1 + y_pre2true + y_pre2false)

        #        loss=T.switch(T.le(epoch,40),-T.log(y_pre2true_softmax),
        #          4*T.nnet.sigmoid(y_pre2true)*T.nnet.sigmoid(y_pre2false)*stimulative*T.log(1+y_pre2true+y_pre2false))

        return loss, stimulative, y_pre2false

    else:
        print "true_dist.ndim != coding_dist.ndim"


def SSL_mutual2(coding_dist, true_dist):
    def set_inf_in2dim(j, coding_dist, true_label_id):
        """
        Search true_label_id==j,and set coding_dist[i][j]="-inf"
        """
        return T.switch(T.eq(j, true_label_id), T.constant(float("-inf")), coding_dist[j])

    def set_inf_in1dim(i, coding_dist, true_label_id):
        # coding_dist[:,label_id] doesn't become "-0.0"
        loss_margin, updates = theano.scan(set_inf_in2dim, \
                                           outputs_info=None, \
                                           sequences=T.arange(coding_dist.shape[1]), \
                                           non_sequences=[coding_dist[i], true_label_id[i]])
        return loss_margin

    if true_dist.ndim == coding_dist.ndim:
        """"""
        coding_dist1 = T.tanh(coding_dist)
        y_pre2true = T.sum(true_dist * T.exp((-coding_dist1)), axis=1)

        # search the true label id
        true_label_id = T.argmax(true_dist, axis=1)
        # persist the false label in coding_dist
        coding_dist_false = (1 - true_dist) * coding_dist1
        # set true label to "-inf"
        coding_dist_true2inf, updates = theano.scan(set_inf_in1dim, \
                                                    outputs_info=None, \
                                                    sequences=T.arange(coding_dist_false.shape[0]), \
                                                    non_sequences=[coding_dist_false, true_label_id])
        # search the max in false label
        coding_dist_true2inf = T.max(coding_dist_true2inf, axis=1)
        y_pre2false = T.exp((coding_dist_true2inf))

        """stimulative"""
        coding_dist = T.nnet.softmax(coding_dist)

        # Calculation: predictioin to true_label
        true_pre = T.sum(true_dist * coding_dist, axis=1)
        #        y_pre2true=T.sum(true_dist * T.exp((3-coding_dist)),axis=1)

        # search the true label id
        true_label_id = T.argmax(true_dist, axis=1)
        # persist the false label in coding_dist
        coding_dist_false = (1 - true_dist) * coding_dist
        # set true label to "-inf"
        coding_dist_true2inf, updates = theano.scan(set_inf_in1dim, \
                                                    outputs_info=None, \
                                                    sequences=T.arange(coding_dist_false.shape[0]), \
                                                    non_sequences=[coding_dist_false, true_label_id])
        # search the max in false label
        coding_dist_true2inf = T.max(coding_dist_true2inf, axis=1)
        #        y_pre2false=T.exp((0.25+coding_dist_true2inf))

        # SSL
        stimulative = 1 + coding_dist_true2inf - true_pre

        #        loss=stimulative*(-T.log(1e-8+true_pre))
        loss = stimulative * T.log(1 + y_pre2true + y_pre2false)

        return loss, y_pre2false, y_pre2true, stimulative

    else:
        print "true_dist.ndim != coding_dist.ndim"