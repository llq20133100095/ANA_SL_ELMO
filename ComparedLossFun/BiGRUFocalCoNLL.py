#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2020.3.19

@author: llq
@function:
    1. realize the BiGRU model
    2. use SDL function
"""

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.utils import floatX
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix
import os
import sys
sys.path.append("..")

from ANASLELMO_in_Conll.dataProcessConll import ELMO_CONLL
from LossFunction import focal_loss

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

theano.config.floatX = "float32"

msg_from = '1182953475@qq.com'
passwd = 'kicjjxxrunufjfej'
msg_to = '1182953475@qq.com'
subject = "Network test1"  # Theme


class Network:

    def __init__(self):
        """network parameters"""
        # the number of unrolled steps of LSTM
        self.num_steps = 107
        # the number of epoch(one epoch=N iterations)
        self.num_epochs = 30
        # the number of class
        self.num_classes = 3
        # the number of GRU units?
        self.cnn_gru_size = 200  # use in cnn
        self.gru_size = 300  # use in gru
        # dropout probability
        self.keep_prob_input = 0.5  # use in input
        self.keep_prob_gru_output = 0.3  # use in gru
        self.keep_prob_cnn = 0.5  # use in cnn
        self.keep_prob_cnn_gur_output = 0.5  # use in output
        # the number of entity pairs of each batch during training or testing
        self.batch_size = 100
        # learning rate
        self.learning_rate = 0.002
        # input shape
        self.embedding_len = 1364 # 1364 or 340
        self.input_shape = (None, self.num_steps, self.embedding_len)
        # mask shape
        self.mask_shape = (None, self.num_steps)
        # All gradients above this will be clipped
        self.grad_clip = 5
        # l2_loss
        self.l2_loss = 1e-2
        # Choose "pi" or "tempens" or "ordinary"
        self.network_type = "ordinary"

        """
        GRU parameter
        """
        self.gate_parameters = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
            b=lasagne.init.Constant(0.))

        self.cell_parameters = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
            # Setting W_cell to None denotes that no cell connection will be used.
            W_cell=None, b=lasagne.init.Constant(0.),
            # By convention, the cell nonlinearity is tanh in an LSTM.
            nonlinearity=lasagne.nonlinearities.tanh)

        """Pi_model and Tempens model"""
        # Ramp learning rate and unsupervised loss weight up during first n epochs.
        self.rampup_length = 60
        # Ramp learning rate and Adam beta1 down during last n epochs.
        self.rampdown_length = 10
        # Unsupervised loss maximum (w_max in paper). Set to 0.0 -> supervised loss only.
        self.scaled_unsup_weight_max = 100.0
        # Maximum learning rate.
        self.learning_rate_max = self.learning_rate
        # Default value.
        self.adam_beta1 = 0.9
        # Target value for Adam beta1 for rampdown.
        self.rampdown_beta1_target = 0.5
        # Total number of labeled inputs (1/10th of this per class). Value 'all' uses all labels.
        self.num_labels = 'all'
        # Ensemble prediction decay constant (\alpha in paper).
        self.prediction_decay = 0.6

        """Save Picture"""
        # # save ACCURACY picture path
        # self.save_picAcc_path = "../result/1/train-test-accuracy.jpg"
        # # save F1 picture path
        # self.save_picF1_path = "../result/1/f1.jpg"
        # self.save_picAllAcc_path = "../result/1/test_all_accuracy.jpg"
        # self.save_picAllRec_path = "../result/1/test_all_recall.jpg"
        # # save train loss picture path
        # self.save_lossTrain_path = "../result/1/train_loss.jpg"
        # # save test loss picture path
        # self.save_lossTest_path = "../result/1/test_loss.jpg"
        # # save result file
        # self.save_result = "../result/1/result.txt"

        # save ACCURACY picture path
        self.save_picAcc_path = "../result/BiGRU/train-test-accuracy.png"
        # save F1 picture path
        self.save_picF1_path = "../result/BiGRU/f1.png"
        self.save_picAllAcc_path = "../result/BiGRU/test_all_accuracy.png"
        self.save_picAllRec_path = "../result/BiGRU/test_all_recall.png"
        # save train loss picture path
        self.save_lossTrain_path = "../result/BiGRU/train_loss.png"
        # save test loss picture path
        self.save_lossTest_path = "../result/BiGRU/test_loss.png"
        # save result file
        self.save_result = "../result/BiGRU/result.txt"

        """loss parameter"""
        self.negative_loss_alpha = np.float32(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.82375]))
        self.negative_loss_lamda = 1
        self.alpha = 0.5
        self.cl_lambda = 1

        """Self Attention"""
        self.input_shape_att = (None, self.embedding_len)  # 1364
        self.attention_size2 = 3

        """ PR-data """
        # self.pr_data = '../result/experiment_PR/1/ANA-SL-ElAtBiGRU_conll04.npz'
        self.pr_data = '../result/experiment_PR/BiGRU_SDL_conll04.npz'

    def bulit_gru(self, input_var=None, mask_var=None, input_root=None, input_e1=None, input_e2=None):
        """
        Bulit the GRU network
        """
        # inputlayer
        l_in = lasagne.layers.InputLayer(shape=self.input_shape, input_var=input_var, name="l_in")

        # mask layer
        l_mask = lasagne.layers.InputLayer(shape=self.mask_shape, input_var=mask_var, name="l_mask")

        # inpute dropout
        l_input_drop = lasagne.layers.DropoutLayer(l_in, p=self.keep_prob_input)

        """main bi-gru"""
        # Two GRU forward
        l_gru_forward = lasagne.layers.GRULayer( \
            l_input_drop, num_units=self.gru_size, grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters, updategate=self.gate_parameters,
            hidden_update=self.cell_parameters, learn_init=True, mask_input=l_mask,
            only_return_final=False, name="l_gru_forward1")

        l_gru_forward = lasagne.layers.GRULayer( \
            l_gru_forward, num_units=self.gru_size, grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters, updategate=self.gate_parameters,
            hidden_update=self.cell_parameters, learn_init=True, mask_input=l_mask,
            only_return_final=True, name="l_gru_forward2")

        # Two GRU backward
        l_gru_backward = lasagne.layers.GRULayer( \
            l_input_drop, num_units=self.gru_size, grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters, updategate=self.gate_parameters,
            hidden_update=self.cell_parameters, learn_init=True, mask_input=l_mask,
            only_return_final=False, backwards=True, name="l_gru_backward1")

        l_gru_backward = lasagne.layers.GRULayer( \
            l_gru_backward, num_units=self.gru_size, grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters, updategate=self.gate_parameters,
            hidden_update=self.cell_parameters, learn_init=True, mask_input=l_mask,
            only_return_final=True, backwards=True, name="l_gru_backward2")

        # Merge forward layers and backward layers
        l_merge = lasagne.layers.ElemwiseSumLayer([l_gru_forward, l_gru_backward])

        # output dropout
        l_merge_drop = lasagne.layers.DropoutLayer(l_merge, p=self.keep_prob_gru_output)

        l_merge_fc = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
            l_merge_drop,
            num_units=250,
            W=lasagne.init.GlorotUniform(gain=1.0), b=lasagne.init.Constant(1.),
            nonlinearity=lasagne.nonlinearities.selu))

        l_out_margin = lasagne.layers.DenseLayer(
            l_merge_fc,
            num_units=self.num_classes,
            W=lasagne.init.GlorotUniform(gain=1.0), b=lasagne.init.Constant(1.),
            nonlinearity=lasagne.nonlinearities.softmax)

        #        return l_out_margin,l_in,l_mask,alphas,l_self_att
        return l_out_margin, l_in, l_mask, l_mask, l_merge_drop

    def mask(self, mask_var, batch_size):
        """
        mask input:it's size is (batch_size,None)
        When mask[i,j]=1,so it can input.
        When mask[i,j]=0,so it can't input.
        """
        mark_input = np.int32(np.zeros((batch_size, self.num_steps)))
        sentence = 0
        for length in mask_var:
            mark_input[sentence] = np.concatenate((np.ones((1, length)), np.zeros((1, self.num_steps - length))),
                                                  axis=1)
            sentence = sentence + 1
        return mark_input

    def adversarial(self, loss, embedding):
        """
        Adversarial training
        """
        perturb = T.grad(loss, embedding)

        # reciprocal in l2 normal
        perturb_rec = 1 / T.sqrt(T.sum(perturb ** 2))
        perturb = 0.01 * perturb * perturb_rec
        embedding = embedding + perturb
        return embedding

    def save_plt(self, x, y, label, title, ylabel_name, save_path, twice=False, x2=None, y2=None, label2=None,
                 showflag=False):
        """
        Save picture:
            1.Train Accuracy
            2.Test Accuracy
            3.Test F1-Score
            4.Train loss
            5.Test loss
        """
        plt.figure(figsize=(10, 10))
        plt.title(title)
        plt.grid()  # open grid
        plt.xlabel('number epoch')
        plt.ylabel(ylabel_name)
        # print two curve
        if twice:
            plt.plot(x, y, label=label)
            plt.legend(loc='best')
            plt.plot(x2, y2, label=label2)
            plt.legend(loc='best')
        else:
            plt.plot(x, y, label=label)
            plt.legend(loc='best')

        plt.savefig(save_path)
        if showflag:
            plt.show()
        plt.close()

    def precision_recall_f1(self, con_mat):
        """
        Compute the top1-precision
        """
        precision = []
        recall = []
        each_precision = 0.0
        each_recall = 0.0
        for i in range(self.num_classes):
            try:
                each_precision = float(con_mat[i][i]) / float(np.sum(con_mat[:, i]))
            except:
                each_precision = 0.0
            try:
                each_recall = float(con_mat[i][i]) / float(np.sum(con_mat[i, :]))
            except:
                each_recall = 0.0

            precision.append(each_precision)
            recall.append(each_recall)

        precision_mean = np.mean(precision)
        recall_mean = np.mean(recall)
        f1 = (2 * precision_mean * recall_mean) / (precision_mean + recall_mean)
        return precision_mean, recall_mean, f1

    def rampup(self, epoch):
        """
        Training utils.
        """
        if epoch < self.rampup_length:
            p = max(0.0, float(epoch)) / float(self.rampup_length)
            p = 1.0 - p
            return math.exp(-p * p * 5.0)
        else:
            return 1.0

    def rampdown(self, epoch):
        if epoch >= (self.num_epochs - self.rampdown_length):
            ep = (epoch - (self.num_epochs - self.rampdown_length)) * 0.5
            return math.exp(-(ep * ep) / self.rampdown_length)
        else:
            return 1.0


if __name__ == "__main__":
    """
    1.Loading data:training data and test data
    """
    print("Loading data")
    elmo_conll = ELMO_CONLL()
    start_time = time.time()

    """ 2. load the dict word2vec """
    elmo_conll.dict_word2vec()
    elmo_conll.label2id_init()
    print("load the dict word2vec: %f s" % (time.time() - start_time))

    """ 3.load the glove embedding """
    # traing_word_pos_vec3D: training data
    train_word_pos_vec3D, train_pos_vec, train_sen_length, train_sen_list2D = \
        elmo_conll.embedding_lookup(elmo_conll.train_sen_store_filename, elmo_conll.train_e1_e2_pos_filename, elmo_conll.train_sen_number)
    train_sen_length = np.int32(np.array(train_sen_length))
    print("load the train glove embedding: %f s" % (time.time() - start_time))

    # testing_word_pos_vec3D: testing data
    test_word_pos_vec3D, test_pos_vec, test_sen_length, test_sen_list2D = \
        elmo_conll.embedding_lookup(elmo_conll.test_sen_store_filename, elmo_conll.test_e1_e2_pos_filename, elmo_conll.test_sen_number)
    test_sen_length = np.int32(np.array(test_sen_length))
    print("load the test glove embedding: %f s" % (time.time() - start_time))

    """ 4.merge the all embedding """
    train_word_pos_vec3D = elmo_conll.merge_glove_elmo(train_word_pos_vec3D, train_pos_vec, elmo_conll.train_elmo_file)
    # train_word_pos_vec3D = np.concatenate((train_word_pos_vec3D, train_pos_vec), axis=2)
    train_word_pos_vec3D = np.float32(train_word_pos_vec3D)
    del train_pos_vec
    test_word_pos_vec3D = elmo_conll.merge_glove_elmo(test_word_pos_vec3D, test_pos_vec, elmo_conll.test_elmo_file)
    # test_word_pos_vec3D = np.concatenate((test_word_pos_vec3D, test_pos_vec), axis=2)
    test_word_pos_vec3D = np.float32(test_word_pos_vec3D)
    del test_pos_vec
    print("merge the all embedding: %f s" % (time.time() - start_time))

    """ 5.load the label """
    # 4.training label
    train_label = elmo_conll.label2id_in_data(elmo_conll.train_label_store_filename, elmo_conll.train_sen_number)
    train_label = np.int32(train_label)

    # 5.testing label
    test_label = elmo_conll.label2id_in_data(elmo_conll.test_label_store_filename, elmo_conll.test_sen_number)
    test_label = np.int32(test_label)

    """ 6.load the embedding of root, e1 and e2. """
    train_root_embedding, train_e1_embedding, train_e2_embedding = \
        elmo_conll.embedding_looking_root_e1_e2(elmo_conll.e1_sdp_train_file, elmo_conll.e2_sdp_train_file, elmo_conll.train_sen_number, train_sen_list2D, elmo_conll.train_elmo_file)
    train_root_embedding = np.concatenate((train_root_embedding[:, :300], train_root_embedding[:, -40:]), axis=1)
    train_e1_embedding = np.concatenate((train_e1_embedding[:, :300], train_e1_embedding[:, -40:]), axis=1)
    train_e2_embedding = np.concatenate((train_e2_embedding[:, :300], train_e2_embedding[:, -40:]), axis=1)

    test_root_embedding, test_e1_embedding, test_e2_embedding=\
        elmo_conll.embedding_looking_root_e1_e2(elmo_conll.e1_sdp_test_file, elmo_conll.e2_sdp_test_file, elmo_conll.test_sen_number, test_sen_list2D, elmo_conll.test_elmo_file)
    print("load the embedding of root, e1 and e2: %f s" % (time.time() - start_time))
    test_root_embedding = np.concatenate((test_root_embedding[:, :300], test_root_embedding[:, -40:]), axis=1)
    test_e1_embedding = np.concatenate((test_e1_embedding[:, :300], test_e1_embedding[:, -40:]), axis=1)
    test_e2_embedding = np.concatenate((test_e2_embedding[:, :300], test_e2_embedding[:, -40:]), axis=1)

    """ 7.label id value and one-hot """
    label2id = elmo_conll.label2id
    train_label_1hot = elmo_conll.label2id_1hot(train_label, label2id)
    train_label_1hot = np.int32(train_label_1hot)

    test_label_1hot = elmo_conll.label2id_1hot(test_label, label2id)
    test_label_1hot = np.int32(test_label_1hot)

    # 9.mask_train_input:unsupervised.
    # Get in pro_data.mask_train_input()
    # Random shuffle.
    indices = np.arange(len(train_word_pos_vec3D))
    np.random.shuffle(indices)
    train_word_pos_vec3D = train_word_pos_vec3D[indices]
    train_sen_length = train_sen_length[indices]
    train_label_1hot = train_label_1hot[indices]
    train_root_embedding = train_root_embedding[indices]
    train_e1_embedding = train_e1_embedding[indices]
    train_e2_embedding = train_e2_embedding[indices]

    """
    new model
    """
    model = Network()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor3('inputs')
    target_var = T.imatrix('targets')
    mask_var = T.imatrix('mask_layer')
    # Pi model variables:
    if model.network_type == "pi":
        input_b_var = T.tensor3('inputs_b')
        mask_train = T.vector('mask_train')
        unsup_weight_var = T.scalar('unsup_weight')
    elif model.network_type == "tempens":
    # tempens model variables:
        z_target_var = T.matrix('z_targets')
        mask_train = T.vector('mask_train')
        unsup_weight_var = T.scalar('unsup_weight')

    learning_rate_var = T.scalar('learning_rate')
    adam_beta1_var = T.scalar('adam_beta1')

    # negative loss
    negative_loss_alpha = T.fvector("negative_loss_alpha")
    negative_loss_lamda = T.fscalar("negative_loss_lamda")

    # input attention entity and root
    input_root = T.fmatrix("input_root")
    input_e1 = T.fmatrix("input_e1")
    input_e2 = T.fmatrix("input_e2")
    # epoch_att=T.iscalar("epoch_att")

    # loss parameter
    centers = theano.shared(np.float32(np.zeros([model.num_classes, 300])), "centers")

    """
    2.
    Bulit GRU network
    ADAM
    """
    gru_network, l_in, l_mask, l_alphas, l_merge_output = model.bulit_gru(input_var,mask_var,input_root,input_e1,input_e2)

    # mask_train_input: where "1" is pass. where "0" isn't pass.
    mask_train_input = elmo_conll.mask_train_input(train_label, num_labels=model.num_labels)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(gru_network)
    alpha = lasagne.layers.get_output(l_alphas)
    l_split = lasagne.layers.get_output(l_merge_output)

    loss, _, _, _ = focal_loss(prediction, target_var)

    # Pi model loss
    if model.network_type=="pi":
        loss = T.mean(loss * mask_train, dtype=theano.config.floatX)
        train_prediction_b = lasagne.layers.get_output(gru_network, inputs={l_in:input_b_var,l_mask:mask_var}) # Second branch.
        loss += unsup_weight_var * T.mean(lasagne.objectives.squared_error(prediction, train_prediction_b))
#        loss=loss+pi_loss
    elif model.network_type=="tempens":
        #Tempens model loss:
        loss = T.mean(loss * mask_train, dtype=theano.config.floatX)
        loss += unsup_weight_var * T.mean(lasagne.objectives.squared_error(prediction, z_target_var))
    else:
        loss = T.mean(loss, dtype=theano.config.floatX)

    # regularization:L1,L2
    l2_penalty = lasagne.regularization.regularize_network_params(gru_network, lasagne.regularization.l2) * model.l2_loss
    loss = loss + l2_penalty

    prediction_batch = T.max(prediction * target_var, axis=1)
    train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), T.argmax(target_var, axis=1)),
                      dtype=theano.config.floatX)
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(gru_network, trainable=True)
    updates = lasagne.updates.adam(
            loss, params, learning_rate=learning_rate_var, beta1=adam_beta1_var)

    """
    3.test loss and accuracy
    """
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(gru_network, deterministic=True)
    test_loss,_,_,st_loss = focal_loss(prediction, target_var)
    test_loss = T.mean(test_loss,dtype=theano.config.floatX)

    # As a bonus, also create an expression for the classification accuracy:
    test_predicted_classid=T.argmax(test_prediction, axis=1)
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)),
                      dtype=theano.config.floatX)

    """
    4.
    train function 
    test function
    """
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    if model.network_type=="pi":
        train_fn = theano.function([input_var, target_var, mask_var, input_b_var, mask_train, unsup_weight_var, learning_rate_var, adam_beta1_var], [loss,train_acc], updates=updates, on_unused_input='warn')
    elif model.network_type=="tempens":
        train_fn = theano.function([input_var, target_var, mask_var, z_target_var, mask_train, unsup_weight_var, learning_rate_var, adam_beta1_var], [loss,train_acc,prediction], updates=updates, on_unused_input='warn')
    else:
        # train_fn = theano.function([input_var, target_var, mask_var, learning_rate_var, adam_beta1_var, negative_loss_alpha, negative_loss_lamda, input_root, input_e1, input_e2], [loss, train_acc, alpha, l_split, loss_batch, prediction_batch], updates=updates, on_unused_input='warn')
        train_fn = theano.function([input_var, target_var, mask_var, learning_rate_var, adam_beta1_var, negative_loss_alpha, negative_loss_lamda, input_root, input_e1, input_e2], [loss, train_acc, alpha, l_split, prediction_batch], updates=updates, on_unused_input='warn')

    # Compile a second function computing the validation loss and accuracy and F1-score:
    val_fn = theano.function([input_var, target_var, mask_var, negative_loss_alpha, negative_loss_lamda, input_root, input_e1, input_e2], [test_loss, test_acc, test_predicted_classid, test_prediction], on_unused_input='warn')

    """
    5.start train
    """
    # Initial training variables for temporal ensembling.

    if model.network_type == 'tempens':
        ensemble_prediction = np.zeros((len(train_word_pos_vec3D), model.num_classes))
        training_targets = np.zeros((len(train_word_pos_vec3D), model.num_classes))

    scaled_unsup_weight_max = model.scaled_unsup_weight_max
    if model.num_labels != 'all':
        scaled_unsup_weight_max *= 1.0 * model.num_labels / train_label.shape[0]

    # Finally, launch the training loop.
    print("Starting training...")

    email_content=""

    # Train accuracy list
    train_acc_listplt=[]
    # Test accuracy list
    test_acc_listplt=[]

    # f1 list:in picture
    f1_listplt=[]
    precision_listplt=[]
    recall_listplt=[]

    # Max f1
    f1_max=0
    # Max num_epoch in F1-SCORE
    f1_max_num_epochs=0

    # Train loss
    train_loss_listplt=[]
    # Test loss
    test_loss_listplt=[]
    mat_conf_list=[]

    aa_l_gru_epoch=[]
    aa_l_split_epoch=[]

    # We iterate over epochs:
    for epoch in range(model.num_epochs):

        epoch = np.int32(epoch)

        # Evaluate up/down ramps.
        rampup_value = model.rampup(epoch)
        rampdown_value = model.rampdown(epoch)

        learning_rate = rampdown_value * model.learning_rate_max
        adam_beta1 = rampdown_value * model.adam_beta1 + (1.0 - rampdown_value) * model.rampdown_beta1_target

        #unsup_weight_var
        unsup_weight = rampup_value * scaled_unsup_weight_max
        if epoch == 0:
            unsup_weight = 0.0

        # Initialize epoch predictions for temporal ensembling.

        if model.network_type == 'tempens':
            epoch_predictions = np.zeros((len(train_word_pos_vec3D), model.num_classes))
            epoch_execmask = np.zeros(len(train_word_pos_vec3D)) # Which inputs were executed.
            training_targets = floatX(training_targets)

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc=0
        train_pi_loss=0
        train_batches = 0
        start_time = time.time()

        stimulative=0

        # loss_compare: compare the SL and the cross entropy
        predict_batch = []
        batch_i = 0

        # Ordinary model
        for batch in elmo_conll.iterate_minibatches_inputAttRootE1E2(train_word_pos_vec3D, \
            train_label_1hot, train_sen_length, model.batch_size, train_root_embedding, train_e1_embedding, train_e2_embedding, shuffle=True):
            aa_inputs, targets, mask_sen_length, input_root, input_e1, input_e2 = batch
            aa_mark_input = model.mask(mask_sen_length, model.batch_size)
            err, acc, aa_l_gru, aa_l_split, predict_each_batch = \
                train_fn(aa_inputs, targets, aa_mark_input, learning_rate, adam_beta1, model.negative_loss_alpha, model.negative_loss_lamda, input_root, input_e1, input_e2)

            train_err+=err
            train_acc+=acc

            train_batches += 1

            if(batch_i == 0):
                predict_batch = np.array(predict_each_batch)
                batch_i = 1
            else:
                predict_batch = np.concatenate((predict_batch, np.array(predict_each_batch)), axis=0)

        aa_l_gru_epoch.append(aa_l_gru)
        aa_l_split_epoch.append(aa_l_split)

        # train accuracy
        train_acc_listplt.append(train_acc / train_batches * 100)
        # train loss
        train_loss_listplt.append(train_err / train_batches)

        # Each epoch training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        test_predicted_classi_all=np.array([0])

        mark_input = model.mask(test_sen_length, len(test_sen_length))
        err, acc, test_predicted_classid, test_prediction_all = val_fn(test_word_pos_vec3D, test_label_1hot, mark_input, model.negative_loss_alpha, model.negative_loss_lamda, test_root_embedding, test_e1_embedding, test_e2_embedding)

        test_err += err
        test_acc += acc
        test_batches += 1

        test_predicted_classi_all = np.concatenate((test_predicted_classi_all, test_predicted_classid))

        test_predicted_classid = np.array(test_predicted_classi_all[1:])

        # test accuracy
        test_acc_listplt.append(test_acc / test_batches * 100)
        # test loss
        test_loss_listplt.append(test_err / test_batches)

        # confusion matrix
        test_con_mat = confusion_matrix(test_label, test_predicted_classid)
        # computer F1_Score
        precision, recall, f1 = model.precision_recall_f1(test_con_mat)

        f1_listplt.append(f1*100)
        mat_conf_list.append(test_con_mat)

        # max f1-score
        if f1_max < f1:
            f1_max = f1
            f1_max_num_epochs = epoch + 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, model.num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.2f} %".format(train_acc / train_batches * 100))

        # Testing loss and accuracy
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))
        print("  test precision:\t\t{:.2f} %".format(precision*100))
        print("  test recall:\t\t{:.2f} %".format(recall*100))
        print("  test f1:\t\t{:.2f} %".format(f1*100))

        print("  max test f1:\t\t{:.2f} %".format(f1_max*100))
        print("  f1 max num epochs:"+str(f1_max_num_epochs))

        # Email content
        email_content=email_content+"Epoch {} of {} took {:.3f}s".format(
            epoch + 1, model.num_epochs, time.time() - start_time)+"\n"
        email_content=email_content+"  training loss:\t\t{:.6f}".format(train_err / train_batches)+"\n"
        email_content=email_content+"  training accuracy:\t\t{:.2f} %".format(train_acc / train_batches * 100)+"\n"

        email_content=email_content+"  test loss:\t\t\t{:.6f}".format(test_err / test_batches)+"\n"
        email_content=email_content+"  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100)+"\n"
        email_content=email_content+"  test precision:\t\t{:.2f} %".format(precision*100)+"\n"
        email_content=email_content+"  test recall:\t\t{:.2f} %".format(recall*100)+"\n"
        email_content=email_content+"  test f1:\t\t{:.2f} %".format(f1*100)+"\n"

        email_content=email_content+"  max test f1:\t\t{:.2f} %".format(f1_max*100)+"\n"

        email_content=email_content+"  f1 max num epochs:"+str(f1_max_num_epochs)+"\n"

        # each 50 epoches,save picture
        if(epoch%10==0 and epoch!=0):
            num_epochs=range(epoch+1)
            num_epochs=[i+1 for i in num_epochs]
            #save Accuracy picture(train and test)
            model.save_plt(x=num_epochs,y=test_acc_listplt,label='test accuracy',title="Train And Test Accuracy",ylabel_name="accuracy(%)",save_path=model.save_picAcc_path,twice=True,x2=num_epochs,y2=train_acc_listplt,label2='train accuracy',showflag=False)
            #save f1 picture
            model.save_plt(num_epochs,f1_listplt,'f1',"F1","f1(%)",model.save_picF1_path,showflag=False)
            #save loss picture(train)
            model.save_plt(num_epochs,train_loss_listplt,'train loss',"Train Loss","loss",model.save_lossTrain_path,showflag=False)
            #save loss picture(test)
            model.save_plt(num_epochs,test_loss_listplt,'test loss',"Test Loss","loss",model.save_lossTest_path,showflag=False)

        # save the prediction
        save_data = dict(
            testing_label_1hot=test_label_1hot,
            test_prediction_all=test_prediction_all)
        np.savez(model.pr_data, **save_data)

    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    print("  test precision:\t\t{:.2f} %".format(precision*100))
    print("  test recall:\t\t{:.2f} %".format(recall*100))
    print("  test f1:\t\t{:.2f} %".format(f1*100))
    print("  max test f1:\t\t{:.2f} %".format(f1_max*100))
    print("  f1 max num epochs:"+str(f1_max_num_epochs))


#    #save the "testing_label" and "test_predicted_classid"
#    ture_predicted_save=np.concatenate((np.reshape(test_predicted_classid,(len(test_predicted_classid),1)),np.reshape(testing_label,(len(testing_label),1))),axis=1)
#    ture_pre_save_txt=open("../testResult/ture_predicted_save.txt","w")
#    for i in range(len(ture_predicted_save)):
#        ture_pre_save_txt.write(str(ture_predicted_save[i][0])+" "+str(ture_predicted_save[i][1]))
#        ture_pre_save_txt.write("\n")
#    ture_pre_save_txt.close()

    """
    8.picture display
    """
    num_epochs=range(model.num_epochs)
    num_epochs=[i+1 for i in num_epochs]
    # save Accuracy picture(train and test)
    model.save_plt(x=num_epochs,y=test_acc_listplt,label='test accuracy',title="Train And Test Accuracy",ylabel_name="accuracy(%)",save_path=model.save_picAcc_path,twice=True,x2=num_epochs,y2=train_acc_listplt,label2='train accuracy')
    # save F1 picture
    model.save_plt(num_epochs,f1_listplt,'f1',"F1","f1(%)",model.save_picF1_path)
    # save loss picture(train)
    model.save_plt(num_epochs,train_loss_listplt,'train loss',"Train Loss","loss",model.save_lossTrain_path)
    # save loss picture(test)
    model.save_plt(num_epochs,test_loss_listplt,'test loss',"Test Loss","loss",model.save_lossTest_path)


    """
    9.save result
    """
    # Email content
    email_content=email_content+"Final results:"+"\n"
    email_content=email_content+"  test loss:\t\t\t{:.6f}".format(test_err / test_batches)+"\n"
    email_content=email_content+"  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100)+"\n"
    email_content=email_content+"  test precision:\t\t{:.2f} %".format(precision*100)+"\n"
    email_content=email_content+"  test f1:\t\t{:.2f} %".format(f1*100)+"\n"
    email_content=email_content+"  max test f1:\t\t{:.2f} %".format(f1_max*100)+"\n"
    email_content=email_content+"  f1 max num epochs:"+str(f1_max_num_epochs)+"\n"

    result_file=open(model.save_result,"w")
    result_file.write("num_steps="+str(model.num_steps)+"\n")
    result_file.write("num_epochs="+str(model.num_epochs)+"\n")
    result_file.write("num_classes="+str(model.num_classes)+"\n")
    result_file.write("cnn_gru_size="+str(model.cnn_gru_size)+"\n")
    result_file.write("gru_size="+str(model.gru_size)+"\n")
    result_file.write("keep_prob_input="+str(model.keep_prob_input)+"\n")
    result_file.write("keep_prob_gru_output="+str(model.keep_prob_gru_output)+"\n")
    result_file.write("keep_prob_cnn="+str(model.keep_prob_cnn)+"\n")
    result_file.write("keep_prob_cnn_gur_output="+str(model.keep_prob_cnn_gur_output)+"\n")
    result_file.write("batch_size="+str(model.batch_size)+"\n")
    result_file.write("learning_rate="+str(model.learning_rate)+"\n")
    result_file.write("network_type="+str(model.network_type)+"\n")
    result_file.write("\n")
    result_file.write("PI MODEL or Tempens model"+"\n")
    result_file.write("rampup_length="+str(model.rampup_length)+"\n")
    result_file.write("rampdown_length="+str(model.rampdown_length)+"\n")
    result_file.write("scaled_unsup_weight_max="+str(model.scaled_unsup_weight_max)+"\n")
    result_file.write("learning_rate_max="+str(model.learning_rate_max)+"\n")
    result_file.write("adam_beta1="+str(model.adam_beta1)+"\n")
    result_file.write("rampdown_beta1_target="+str(model.rampdown_beta1_target)+"\n")
    result_file.write("num_labels="+str(model.num_labels)+"\n")
    result_file.write("prediction_decay="+str(model.prediction_decay)+"\n")
    result_file.write("\n")
    result_file.write("f1_max="+str(f1_max)+",f1_max_num_epochs="+str(f1_max_num_epochs)+"\n")
    result_file.write("\n")
    result_file.write(email_content)
    result_file.close()

    """
    10.send result to my email
    """
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = msg_from
    msg['To'] = msg_to

    # Text Content
    msg.attach(MIMEText(email_content, 'plain', 'utf-8'))

    # Construct attachment 1 and send the picture file in the current directory
    att1 = MIMEText(open(model.save_picAcc_path, 'rb').read(), 'base64', 'utf-8')
    att1["Content-Type"] = 'application/octet-stream'
    # filename in email display
    att1["Content-Disposition"] = 'attachment; filename="train and test accuracy.jpg"'
    msg.attach(att1)

    # Construct attachment 2 and send the picture file in the current directory
    att1 = MIMEText(open(model.save_picF1_path, 'rb').read(), 'base64', 'utf-8')
    att1["Content-Type"] = 'application/octet-stream'
    # filename in email display
    att1["Content-Disposition"] = 'attachment; filename="f1-score.jpg"'
    msg.attach(att1)
    try:
        s = smtplib.SMTP_SSL("smtp.qq.com",465)
        s.login(msg_from, passwd)
        s.sendmail(msg_from, msg_to, msg.as_string())
        print("发送成功")
    except s.SMTPException,e:
        print("发送失败")
    finally:
        s.quit()