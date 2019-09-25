#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:43:00 2018

KBP dataset in relation classification

@author: llq
"""
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.utils import floatX
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import precision_recall_fscore_support, f1_score, classification_report, confusion_matrix

from ANASLELMO_in_Conll.dataProcessConll import ELMO_CONLL
from SdpNetwork.CustomLayers import SplitInLeft, SplitInRight, SplitInGlobal, HighwayNetwork1D, HighwayNetwork2D, \
    MarginLossLayer
from SdpNetwork.CustomLoss import Negative_loss, Margin_loss

theano.config.floatX = "float32"

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

msg_from = '1182953475@qq.com'
passwd = 'kicjjxxrunufjfej'
msg_to = '1182953475@qq.com'
subject = "Network test1"  # Theme


class Network:

    def __init__(self):
        """network parameters"""
        # the number of unrolled steps of LSTM
        self.num_steps = 82
        # the number of epoch(one epoch=N iterations)
        self.num_epochs = 130
        # the number of class
        self.num_classes = 40
        # the number of GRU units?
        self.cnn_gru_size = 340  # use in cnn
        self.gru_size = 300  # use in gru
        # dropout probability
        self.keep_prob_input = 0.4  # use in input
        self.keep_prob_gru_output = 0.5  # use in gru
        self.keep_prob_cnn = 0.5  # use in cnn
        self.keep_prob_cnn_gur_output = 0.5  # use in output
        # the number of entity pairs of each batch during training or testing
        self.batch_size = 100
        # learning rate
        self.learning_rate = 0.001
        # input shape
        self.input_shape = (None, 82, 340)
        # mask shape
        self.mask_shape = (None, 82)
        # All gradients above this will be clipped
        self.grad_clip = 0.5
        # l2_loss
        self.l2_loss = 1e-4
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
        self.rampup_length = 30
        # Ramp learning rate and Adam beta1 down during last n epochs.
        self.rampdown_length = 30
        # Unsupervised loss maximum (w_max in paper). Set to 0.0 -> supervised loss only.
        self.scaled_unsup_weight_max = 100.0
        # Maximum learning rate.
        self.learning_rate_max = 0.001
        # Default value.
        self.adam_beta1 = 0.9
        # Target value for Adam beta1 for rampdown.
        self.rampdown_beta1_target = 0.5
        # Total number of labeled inputs (1/10th of this per class). Value 'all' uses all labels.
        self.num_labels = 'all'
        # Ensemble prediction decay constant (\alpha in paper).
        self.prediction_decay = 0.6

        """Save Picture"""
        # save ACCURACY picture path
        self.save_picAcc_path = "../result/train-test-accuracy.jpg"
        # save F1 picture path
        self.save_picF1_path = "../result/f1.jpg"
        self.save_picAllAcc_path = "../result/test all accuracy.jpg"
        self.save_picAllRec_path = "../result/test all recall.jpg"
        # save train loss picture path
        self.save_lossTrain_path = "../result/train-loss.jpg"
        # save test loss picture path
        self.save_lossTest_path = "../result/test-loss.jpg"
        # save result file
        self.save_result = "../result/result.txt"

        """Negtive loss"""
        self.negative_loss_alpha = np.float32(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.82375]))
        self.negative_loss_lamda = 1

    def bulit_gru(self, input_var=None, mask_var=None, sen_length=None):
        """
        Bulit the GRU network
        """
        # inputlayer
        l_in = lasagne.layers.InputLayer(shape=self.input_shape, input_var=input_var, name="l_in")

        # mask layer
        l_mask = lasagne.layers.InputLayer(shape=self.mask_shape, input_var=mask_var, name="l_mask")

        # inpute dropout
        l_input_drop = lasagne.layers.DropoutLayer(l_in, p=self.keep_prob_input)

        """
        CNN
        """
        # the length of sentences
        l_sen_length = lasagne.layers.InputLayer(shape=(None, 1), input_var=sen_length, name="l_sen_length")

        # split the global SDP
        l_split_global_sdp = SplitInGlobal((l_sen_length, l_input_drop))
        l_split_global_sdp = lasagne.layers.ReshapeLayer(l_split_global_sdp, ([0], 1, [1], [2]))

        l_global_sdp_cnn = lasagne.layers.Conv2DLayer(
            l_split_global_sdp, num_filters=500, filter_size=(3, self.cnn_gru_size),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        # MAX-pooling in global SDP
        l_global_sdp_maxpooling = lasagne.layers.GlobalPoolLayer(l_global_sdp_cnn, pool_function=T.max)

        # output dropout
        l_sdp_drop = lasagne.layers.DropoutLayer(l_global_sdp_maxpooling, p=self.keep_prob_cnn)

        l_out_den = lasagne.layers.DenseLayer( \
            l_sdp_drop, num_units=200, \
            nonlinearity=lasagne.nonlinearities.rectify)

        #        #output dropout
        #        l_merge_drop=lasagne.layers.DropoutLayer(l_merge,p=self.keep_prob_gru_output)

        # Finally, we'll add the fully-connected output layer, of 10 softmax units:
        l_out = lasagne.layers.DenseLayer( \
            l_out_den, num_units=self.num_classes, \
            nonlinearity=lasagne.nonlinearities.softmax)

        return l_out, l_in, l_mask, l_out_den, l_sdp_drop

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

    def save_plt(self, x, y, label, title, ylabel_name, save_path, twice=False, x2=None, y2=None, label2=None,
                 showflag=True):
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
    # train_word_pos_vec3D = elmo_conll.merge_glove_elmo(train_word_pos_vec3D, train_pos_vec, elmo_conll.train_elmo_file)
    train_word_pos_vec3D = np.concatenate((train_word_pos_vec3D, train_pos_vec), axis=2)
    train_word_pos_vec3D = np.float32(train_word_pos_vec3D)
    del train_pos_vec
    # test_word_pos_vec3D = elmo_conll.merge_glove_elmo(test_word_pos_vec3D, test_pos_vec, elmo_conll.test_elmo_file)
    test_word_pos_vec3D = np.concatenate((test_word_pos_vec3D, test_pos_vec), axis=2)
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

    #    #Left sdp length
    #    left_sdp_length=T.imatrix('left_sdp_length')
    # Sentences length
    sen_length = T.imatrix('sen_length')

    # negative loss
    negative_loss_alpha = T.fvector("negative_loss_alpha")
    negative_loss_lamda = T.fscalar("negative_loss_lamda")

    """
    2.
    Bulit GRU network
    ADAM
    """
    gru_network, l_in, l_mask, l_gru_forward, l_split_cnn = model.bulit_gru(input_var, mask_var, sen_length)

    # mask_train_input: where "1" is pass. where "0" isn't pass.
    mask_train_input = elmo_conll.mask_train_input(train_label, num_labels=model.num_labels)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(gru_network)
    l_gru = lasagne.layers.get_output(l_gru_forward)
    l_split = lasagne.layers.get_output(l_split_cnn)

    #    loss,_,_ = Margin_loss(prediction, target_var)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)

    # Pi model loss
    if model.network_type == "pi":
        loss = T.mean(loss * mask_train, dtype=theano.config.floatX)
        train_prediction_b = lasagne.layers.get_output(gru_network,
                                                       inputs={l_in: input_b_var, l_mask: mask_var})  # Second branch.
        loss += unsup_weight_var * T.mean(lasagne.objectives.squared_error(prediction, train_prediction_b))
    #        loss=loss+pi_loss
    elif model.network_type == "tempens":
        # Tempens model loss:
        loss = T.mean(loss * mask_train, dtype=theano.config.floatX)
        loss += unsup_weight_var * T.mean(lasagne.objectives.squared_error(prediction, z_target_var))
    else:
        loss = T.mean(loss, dtype=theano.config.floatX)

    # regularization:L1,L2
    l2_penalty = lasagne.regularization.regularize_network_params(gru_network,
                                                                  lasagne.regularization.l2) * model.l2_loss
    loss = loss + l2_penalty

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
    #    test_loss,_,_ = Margin_loss(test_prediction,target_var)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = T.mean(test_loss, dtype=theano.config.floatX)

    # As a bonus, also create an expression for the classification accuracy:
    # ????????????????????
    test_predicted_classid = T.argmax(test_prediction, axis=1)
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)),
                      dtype=theano.config.floatX)

    """
    4.
    train function 
    test function
    """
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    if model.network_type == "pi":
        train_fn = theano.function(
            [input_var, target_var, mask_var, input_b_var, mask_train, unsup_weight_var, learning_rate_var,
             adam_beta1_var], [loss, train_acc], updates=updates, on_unused_input='warn')
    elif model.network_type == "tempens":
        train_fn = theano.function(
            [input_var, target_var, mask_var, z_target_var, mask_train, unsup_weight_var, learning_rate_var,
             adam_beta1_var], [loss, train_acc, prediction], updates=updates, on_unused_input='warn')
    else:
        train_fn = theano.function(
            [input_var, target_var, mask_var, learning_rate_var, adam_beta1_var, negative_loss_alpha,
             negative_loss_lamda, sen_length], [loss, train_acc, l_gru, l_split], updates=updates,
            on_unused_input='warn')

    # Compile a second function computing the validation loss and accuracy and F1-score:
    val_fn = theano.function([input_var, target_var, mask_var, negative_loss_alpha, negative_loss_lamda, sen_length],
                             [test_loss, test_acc, test_predicted_classid, test_prediction], on_unused_input='warn')

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

    email_content = ""

    # Train accuracy list
    train_acc_listplt = []
    # Test accuracy list
    test_acc_listplt = []

    # f1 list:in picture
    f1_listplt = []
    precision_listplt = []
    recall_listplt = []

    # Max f1
    f1_max = 0
    # Max num_epoch in F1-SCORE
    f1_max_num_epochs = 0

    # Train loss
    train_loss_listplt = []
    # Test loss
    test_loss_listplt = []
    mat_conf_list = []

    # We iterate over epochs:
    for epoch in range(model.num_epochs):

        # Evaluate up/down ramps.
        rampup_value = model.rampup(epoch)
        rampdown_value = model.rampdown(epoch)

        learning_rate = rampdown_value * model.learning_rate_max
        adam_beta1 = rampdown_value * model.adam_beta1 + (1.0 - rampdown_value) * model.rampdown_beta1_target

        # unsup_weight_var
        unsup_weight = rampup_value * scaled_unsup_weight_max
        if epoch == 0:
            unsup_weight = 0.0

        # Initialize epoch predictions for temporal ensembling.

        if model.network_type == 'tempens':
            epoch_predictions = np.zeros((len(train_word_pos_vec3D), model.num_classes))
            epoch_execmask = np.zeros(len(train_word_pos_vec3D))  # Which inputs were executed.
            training_targets = floatX(training_targets)

            # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_pi_loss = 0
        train_batches = 0
        start_time = time.time()

        # Ordinary model
        for batch in elmo_conll.iterate_minibatches(train_word_pos_vec3D, \
                                                  train_label_1hot, train_sen_length, model.batch_size,
                                                  shuffle=True):
            aa_inputs, targets, mask_sen_length = batch
            aa_mark_input = model.mask(mask_sen_length, model.batch_size)
            err, acc, aa_l_gru, aa_l_split = train_fn(aa_inputs, targets, aa_mark_input, learning_rate, adam_beta1,
                                                      model.negative_loss_alpha, model.negative_loss_lamda,
                                                      np.reshape(mask_sen_length, (-1, 1)))
            train_err += err
            train_acc += acc
            train_batches += 1

        # train accuracy
        train_acc_listplt.append(train_acc / train_batches * 100)
        # train loss
        train_loss_listplt.append(train_err / train_batches)

        # Each epoch training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        test_predicted_classi_all = np.array([0])
        test_prediction_all = np.reshape(np.array([0.0] * model.num_classes), (1, -1))

        mark_input = model.mask(test_sen_length, len(test_sen_length))
        err, acc, test_predicted_classid, test_prediction_all = val_fn(test_word_pos_vec3D, test_label_1hot, mark_input, model.negative_loss_alpha, model.negative_loss_lamda, np.reshape(test_sen_length, (-1, 1)))

        # for batch in elmo_conll.iterate_minibatches(test_word_pos_vec3D, \
        #                                           test_label_1hot, test_sen_length, model.batch_size,
        #                                           shuffle=False):
        #     inputs, targets, mask_sen_length = batch
        #     mark_input = model.mask(mask_sen_length, model.batch_size)
        #     err, acc, test_predicted_classid, test_prediction = val_fn(inputs, targets, mark_input,
        #                                                                model.negative_loss_alpha,
        #                                                                model.negative_loss_lamda,
        #                                                                np.reshape(mask_sen_length, (-1, 1)))
        test_err += err
        test_acc += acc
        test_batches += 1

        test_predicted_classi_all = np.concatenate((test_predicted_classi_all, test_predicted_classid))

        test_prediction_all = np.concatenate((test_prediction_all, test_prediction), axis=0)

        test_predicted_classid = np.array(test_predicted_classi_all[1:])
        # test accuracy
        test_acc_listplt.append(test_acc / test_batches * 100)
        # test loss
        test_loss_listplt.append(test_err / test_batches)

        # update "ensemble_prediction(Z)" and "training_targets"
        if model.network_type == 'tempens':
            # Basic mode.
            ensemble_prediction = (model.prediction_decay * ensemble_prediction) + (
                        1.0 - model.prediction_decay) * epoch_predictions
            training_targets = ensemble_prediction / (1.0 - model.prediction_decay ** ((epoch - 0) + 1.0))

        # confusion matrix
        test_con_mat = confusion_matrix(test_label, test_predicted_classid)
        # computer F1_Score
        precision, recall, f1 = model.precision_recall_f1(test_con_mat)

        f1_listplt.append(f1 * 100)
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
        print("  test precision:\t\t{:.2f} %".format(precision * 100))
        print("  test recall:\t\t{:.2f} %".format(recall * 100))
        print("  test f1:\t\t{:.2f} %".format(f1 * 100))
        print("  max test f1:\t\t{:.2f} %".format(f1_max * 100))
        print("  f1 max num epochs:" + str(f1_max_num_epochs))

        # Email content
        email_content = email_content + "Epoch {} of {} took {:.3f}s".format(
            epoch + 1, model.num_epochs, time.time() - start_time) + "\n"
        email_content = email_content + "  training loss:\t\t{:.6f}".format(train_err / train_batches) + "\n"
        email_content = email_content + "  training accuracy:\t\t{:.2f} %".format(
            train_acc / train_batches * 100) + "\n"
        email_content = email_content + "  test loss:\t\t\t{:.6f}".format(test_err / test_batches) + "\n"
        email_content = email_content + "  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100) + "\n"
        email_content = email_content + "  test precision:\t\t{:.2f} %".format(precision * 100) + "\n"
        email_content = email_content + "  test recall:\t\t{:.2f} %".format(recall * 100) + "\n"
        email_content = email_content + "  test f1:\t\t{:.2f} %".format(f1 * 100) + "\n"
        email_content = email_content + "  max test f1:\t\t{:.2f} %".format(f1_max * 100) + "\n"
        email_content = email_content + "  f1 max num epochs:" + str(f1_max_num_epochs) + "\n"

        # each 50 epoches,save picture
        if (epoch % 50 == 0 and epoch != 0):
            num_epochs = range(epoch + 1)
            num_epochs = [i + 1 for i in num_epochs]
            # save Accuracy picture(train and test)
            model.save_plt(x=num_epochs, y=test_acc_listplt, label='test accuracy', title="Train And Test Accuracy",
                           ylabel_name="accuracy(%)", save_path=model.save_picAcc_path, twice=True, x2=num_epochs,
                           y2=train_acc_listplt, label2='train accuracy', showflag=False)
            # save f1 picture
            model.save_plt(num_epochs, f1_listplt, 'f1', "F1", "f1(%)", model.save_picF1_path, showflag=False)
            # save loss picture(train)
            model.save_plt(num_epochs, train_loss_listplt, 'train loss', "Train Loss", "loss",
                           model.save_lossTrain_path, showflag=False)
            # save loss picture(test)
            model.save_plt(num_epochs, test_loss_listplt, 'test loss', "Test Loss", "loss", model.save_lossTest_path,
                           showflag=False)

    # save the prediction
    testing_label_1hot = test_label_1hot[:9574]
    test_prediction_all = test_prediction_all[1:9575]
    save_data = dict(
        testing_label_1hot=testing_label_1hot,
        test_prediction_all=test_prediction_all)
    np.savez('../experimentCharts/kbp_experiment/CNN_KBP.npz', **save_data)

    '''
    """
    7.test model
    """
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in kbp_data.iterate_minibatches(testing_word_pos_vec3D, \
      testing_label_1hot, testing_sen_length, model.batch_size, shuffle=False):
        inputs, targets, mask_sen_length = batch
        mark_input=model.mask(mask_sen_length,model.batch_size)
        err, acc, test_predicted_classid = val_fn(inputs, targets, mark_input, model.negative_loss_alpha, model.negative_loss_lamda, np.reshape(mask_sen_length,(-1,1)))
        test_err += err
        test_acc += acc
        test_batches += 1

    #F1 value
    mark_input=model.mask(testing_sen_length,len(testing_sen_length))
    err, acc, test_predicted_classid=val_fn(testing_word_pos_vec3D, testing_label_1hot, mark_input, model.negative_loss_alpha, model.negative_loss_lamda, np.reshape(testing_sen_length,(-1,1)))
    #confusion matrix
    test_con_mat=confusion_matrix(testing_label,test_predicted_classid)
#    #support
#    _,_,_,support=precision_recall_fscore_support(testing_label,test_predicted_classid)   
    #computer F1_Score
    precision,recall,f1=model.precision_recall_f1(test_con_mat)

    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    print("  test precision:\t\t{:.2f} %".format(precision*100))
    print("  test recall:\t\t{:.2f} %".format(recall*100))
    print("  test f1:\t\t{:.2f} %".format(f1*100))
    print("  max test f1:\t\t{:.2f} %".format(f1_max*100))
    print("  f1 max num epochs:"+str(f1_max_num_epochs))

    #save the "testing_label" and "test_predicted_classid"
    ture_predicted_save=np.concatenate((np.reshape(test_predicted_classid,(len(test_predicted_classid),1)),np.reshape(testing_label,(len(testing_label),1))),axis=1)
    ture_pre_save_txt=open("../testResult/ture_predicted_save.txt","w")
    for i in range(len(ture_predicted_save)):
        ture_pre_save_txt.write(str(ture_predicted_save[i][0])+" "+str(ture_predicted_save[i][1]))
        ture_pre_save_txt.write("\n")
    ture_pre_save_txt.close()
    '''

    """
    8.picture display
    """
    num_epochs = range(model.num_epochs)
    num_epochs = [i + 1 for i in num_epochs]
    # save Accuracy picture(train and test)
    model.save_plt(x=num_epochs, y=test_acc_listplt, label='test accuracy', title="Train And Test Accuracy",
                   ylabel_name="accuracy(%)", save_path=model.save_picAcc_path, twice=True, x2=num_epochs,
                   y2=train_acc_listplt, label2='train accuracy')
    # save F1 picture
    model.save_plt(num_epochs, f1_listplt, 'f1', "F1", "f1(%)", model.save_picF1_path)
    # save loss picture(train)
    model.save_plt(num_epochs, train_loss_listplt, 'train loss', "Train Loss", "loss", model.save_lossTrain_path)
    # save loss picture(test)
    model.save_plt(num_epochs, test_loss_listplt, 'test loss', "Test Loss", "loss", model.save_lossTest_path)

    """
    9.send result to my email
    """
    # Email content
    email_content = email_content + "Final results:" + "\n"
    email_content = email_content + "  test loss:\t\t\t{:.6f}".format(test_err / test_batches) + "\n"
    email_content = email_content + "  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100) + "\n"
    email_content = email_content + "  test precision:\t\t{:.2f} %".format(precision * 100) + "\n"
    email_content = email_content + "  test f1:\t\t{:.2f} %".format(f1 * 100) + "\n"
    email_content = email_content + "  max test f1:\t\t{:.2f} %".format(f1_max * 100) + "\n"
    email_content = email_content + "  f1 max num epochs:" + str(f1_max_num_epochs) + "\n"

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
        s = smtplib.SMTP_SSL("smtp.qq.com", 465)
        s.login(msg_from, passwd)
        s.sendmail(msg_from, msg_to, msg.as_string())
        print "发送成功"
    except s.SMTPException, e:
        print "发送失败"
    finally:
        s.quit()

    """
    10.save result
    """
    result_file = open(model.save_result, "w")
    result_file.write("num_steps=" + str(model.num_steps) + "\n")
    result_file.write("num_epochs=" + str(model.num_epochs) + "\n")
    result_file.write("num_classes=" + str(model.num_classes) + "\n")
    result_file.write("cnn_gru_size=" + str(model.cnn_gru_size) + "\n")
    result_file.write("gru_size=" + str(model.gru_size) + "\n")
    result_file.write("keep_prob_input=" + str(model.keep_prob_input) + "\n")
    result_file.write("keep_prob_gru_output=" + str(model.keep_prob_gru_output) + "\n")
    result_file.write("keep_prob_cnn=" + str(model.keep_prob_cnn) + "\n")
    result_file.write("keep_prob_cnn_gur_output=" + str(model.keep_prob_cnn_gur_output) + "\n")
    result_file.write("batch_size=" + str(model.batch_size) + "\n")
    result_file.write("learning_rate=" + str(model.learning_rate) + "\n")
    result_file.write("network_type=" + str(model.network_type) + "\n")
    result_file.write("\n")
    result_file.write("PI MODEL or Tempens model" + "\n")
    result_file.write("rampup_length=" + str(model.rampup_length) + "\n")
    result_file.write("rampdown_length=" + str(model.rampdown_length) + "\n")
    result_file.write("scaled_unsup_weight_max=" + str(model.scaled_unsup_weight_max) + "\n")
    result_file.write("learning_rate_max=" + str(model.learning_rate_max) + "\n")
    result_file.write("adam_beta1=" + str(model.adam_beta1) + "\n")
    result_file.write("rampdown_beta1_target=" + str(model.rampdown_beta1_target) + "\n")
    result_file.write("num_labels=" + str(model.num_labels) + "\n")
    result_file.write("prediction_decay=" + str(model.prediction_decay) + "\n")
    result_file.write("\n")
    result_file.write("f1_max=" + str(f1_max) + ",f1_max_num_epochs=" + str(f1_max_num_epochs) + "\n")
    result_file.write("\n")
    result_file.write(email_content)
    result_file.close()

