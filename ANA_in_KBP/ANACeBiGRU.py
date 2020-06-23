<<<<<<< HEAD
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2020.3.20

@author: llq
@function:
    1. realise the cross entropy in BiGRU
    2. multi-head attention
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
import sys
sys.path.append("..")

from KbpNetwork.kbpProcessData import kbpProcess
from customLayers import Input_keywords_DotLayer
from MultiheadAttention_Op import multihead_attention, multihead_attention_origin

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
        self.num_epochs = 64
        # the number of class
        self.num_classes = 40
        # the number of GRU units?
        self.cnn_gru_size = 200  # use in cnn
        self.gru_size = 300  # use in gru
        # dropout probability
        self.keep_prob_input = 0.3  # use in input
        self.keep_prob_gru_output = 0.3  # use in gru output
        # the number of entity pairs of each batch during training or testing
        self.batch_size = 100
        # learning rate
        self.learning_rate = 0.002
        # input shape
        self.embedding_len = 340
        self.input_shape = (None, 82, self.embedding_len)
        self.input_shape_att = (None, self.embedding_len)
        # mask shape
        self.mask_shape = (None, 82)
        # All gradients above this will be clipped
        self.grad_clip = 5
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
        self.rampup_length = 4
        # Ramp learning rate and Adam beta1 down during last n epochs.
        self.rampdown_length = 30
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
        # save ACCURACY picture path
        self.save_picAcc_path = "../result/train-test-accuracy.png"
        # save F1 picture path
        self.save_picF1_path = "../result/f1.png"
        self.save_picAllAcc_path = "../result/test_all_accuracy.png"
        self.save_picAllRec_path = "../result/test_all_recall.png"
        # save train loss picture path
        self.save_lossTrain_path = "../result/train-loss.png"
        # save test loss picture path
        self.save_lossTest_path = "../result/test-loss.png"
        # save lr path
        self.save_lr_path = "../result/lr.png"
        # save result file
        self.save_result = "../result/result.txt"

        """Negtive loss"""
        self.alpha = 0.5
        self.cl_lambda = 1

        """Self Attention"""
        self.attention_size2 = 2

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

        """Input attention entity and root"""
        # l_in_root = lasagne.layers.InputLayer(shape=self.input_shape_att, input_var=input_root, name="l_in_root")
        # l_in_e1 = lasagne.layers.InputLayer(shape=self.input_shape_att, input_var=input_e1, name="l_in_e1")
        # l_in_e2 = lasagne.layers.InputLayer(shape=self.input_shape_att, input_var=input_e2, name="l_in_e2")
        #
        # # Dot the "input" and "keywords"
        # l_in_root = lasagne.layers.ReshapeLayer(l_in_root, (-1, 1, self.input_shape_att[1]))
        # l_in_e1 = lasagne.layers.ReshapeLayer(l_in_e1, (-1, 1, self.input_shape_att[1]))
        # l_in_e2 = lasagne.layers.ReshapeLayer(l_in_e2, (-1, 1, self.input_shape_att[1]))
        # l_keywords = lasagne.layers.ConcatLayer([l_in_e1, l_in_root, l_in_e2], axis=1, name="l_keywords")

        """main bi-gru"""
        # Two GRU forward
        l_gru_forward = lasagne.layers.GRULayer( \
            l_input_drop, num_units=self.gru_size, grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters, updategate=self.gate_parameters,
            hidden_update=self.cell_parameters, learn_init=True,
            only_return_final=False, name="l_gru_forward1")

        l_gru_forward = lasagne.layers.GRULayer( \
            l_gru_forward, num_units=self.gru_size, grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters, updategate=self.gate_parameters,
            hidden_update=self.cell_parameters, learn_init=True,
            only_return_final=False, name="l_gru_forward2")

        # Two GRU backward
        l_gru_backward = lasagne.layers.GRULayer( \
            l_input_drop, num_units=self.gru_size, grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters, updategate=self.gate_parameters,
            hidden_update=self.cell_parameters, learn_init=True,
            only_return_final=False, backwards=True, name="l_gru_backward1")

        l_gru_backward = lasagne.layers.GRULayer( \
            l_gru_backward, num_units=self.gru_size, grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters, updategate=self.gate_parameters,
            hidden_update=self.cell_parameters, learn_init=True,
            only_return_final=False, backwards=True, name="l_gru_backward2")

        # Merge forward layers and backward layers
        l_merge = lasagne.layers.ElemwiseSumLayer([l_gru_forward, l_gru_backward])

        """auxiliary bi-gru"""
        # l_keywords = lasagne.layers.DropoutLayer(l_keywords, p=self.keep_prob_input)
        #
        # l_aux_gru_forward = lasagne.layers.GRULayer( \
        #     l_keywords, num_units=self.gru_size, grad_clipping=self.grad_clip,
        #     resetgate=self.gate_parameters, updategate=self.gate_parameters,
        #     hidden_update=self.cell_parameters, learn_init=True,
        #     only_return_final=True, name="l_aux_gru_forward")
        #
        # l_aux_gru_backward = lasagne.layers.GRULayer( \
        #     l_keywords, num_units=self.gru_size, grad_clipping=self.grad_clip,
        #     resetgate=self.gate_parameters, updategate=self.gate_parameters,
        #     hidden_update=self.cell_parameters, learn_init=True,
        #     only_return_final=True, backwards=True, name="l_aux_gru_backward")
        #
        # # Merge forward layers and backward layers
        # l_aux_merge = lasagne.layers.ElemwiseSumLayer([l_aux_gru_forward, l_aux_gru_backward])

        """multi-head attention"""
        # l_att = multihead_attention([l_keywords, l_in, l_merge, l_aux_merge], name="multihead_attention")

        for i in range(4):
            l_merge = multihead_attention_origin([l_merge, l_merge, l_merge], name="multihead_attention")

        # output dropout
        l_merge_drop = lasagne.layers.DropoutLayer(l_merge, p=self.keep_prob_gru_output)

        l_merge_fc = lasagne.layers.DenseLayer(
            l_merge_drop,
            num_units=250,
            W=lasagne.init.GlorotUniform(gain=1.0), b=lasagne.init.Constant(1.),
            nonlinearity=lasagne.nonlinearities.selu)

        l_out_margin = lasagne.layers.DenseLayer(
            l_merge_fc,
            num_units=self.num_classes,
            nonlinearity=lasagne.nonlinearities.softmax)

        return l_out_margin, l_in, l_mask, l_merge, l_merge

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

    def send_pic(self, pic_path, msg, pic_name):
        # Construct attachment 1 and send the picture file in the current directory
        att1 = MIMEText(open(pic_path, 'rb').read(), 'base64', 'utf-8')
        att1["Content-Type"] = 'application/octet-stream'
        # filename in email display
        att1["Content-Disposition"] = 'attachment; filename="' + pic_name + '"'
        msg.attach(att1)


if __name__ == "__main__":

    """
    1.Loading data:training data and test data
    """
    print("Loading data")
    # 1.Class:Process_data() and init the dict_word_vec
    kbp_data = kbpProcess()
    kbp_data.dict_word2vec()
    kbp_data.label2id_init()

    # 2.traing_word_pos_vec3D:training data
    training_word_pos_vec3D, training_sen_length, train_sen_list2D = \
        kbp_data.embedding_lookup(kbp_data.train_sen_store_filename, \
                                  kbp_data.training_e1_e2_pos_filename, kbp_data.training_sen_number)
    training_word_pos_vec3D = np.float32(training_word_pos_vec3D)
    training_sen_length = np.int32(np.array(training_sen_length))

    # 3.testing_word_pos_vec3D:testing data
    testing_word_pos_vec3D, testing_sen_length, test_sen_list2D = kbp_data.embedding_lookup(
        kbp_data.test_sen_store_filename, \
        kbp_data.testing_e1_e2_pos_filename, kbp_data.testing_sen_number)
    testing_word_pos_vec3D = np.float32(testing_word_pos_vec3D)
    testing_sen_length = np.int32(np.array(testing_sen_length))

    # 4.training label:8000
    #  mask_train:unsupervised
    training_label = kbp_data.label2id_in_data(kbp_data.train_label_store_filename, \
                                               kbp_data.training_sen_number)
    training_label = np.int32(training_label)

    # 5.testing label:10825
    testing_label = kbp_data.label2id_in_data(kbp_data.test_label_store_filename, \
                                              kbp_data.testing_sen_number)
    testing_label = np.int32(testing_label)

    # 6.label id value: Change the label to id.And 10 classes number(0-9)
    label2id = kbp_data.label2id

    # 7.One-hot encode
    training_label_1hot = kbp_data.label2id_1hot(training_label, label2id)
    training_label_1hot = np.int32(training_label_1hot)

    testing_label_1hot = kbp_data.label2id_1hot(testing_label, label2id)
    testing_label_1hot = np.int32(testing_label_1hot)

    # 8.embedding root,e1 and e2
    train_root_embedding, train_e1_embedding, train_e2_embedding = \
        kbp_data.embedding_looking_root_e1_e2(kbp_data.e1_sdp_train_file, kbp_data.e2_sdp_train_file,
                                              kbp_data.training_sen_number, train_sen_list2D)

    test_root_embedding, test_e1_embedding, test_e2_embedding = \
        kbp_data.embedding_looking_root_e1_e2(kbp_data.e1_sdp_test_file, kbp_data.e2_sdp_test_file,
                                              kbp_data.testing_sen_number, test_sen_list2D)

    # 9.mask_train_input:unsupervised.
    # Get in pro_data.mask_train_input()
    # Random shuffle.
    indices = np.arange(len(training_word_pos_vec3D))
    np.random.shuffle(indices)
    training_word_pos_vec3D = training_word_pos_vec3D[indices]
    training_sen_length = training_sen_length[indices]
    training_label_1hot = training_label_1hot[indices]
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

    # input attention entity and root
    input_root = T.fmatrix("input_root")
    input_e1 = T.fmatrix("input_e1")
    input_e2 = T.fmatrix("input_e2")
    epoch_att = T.iscalar("epoch_att")

    # loss parameter
    centers = theano.shared(np.float32(np.zeros([model.num_classes, 250])), "centers")

    """
    2.
    Bulit GRU network
    ADAM
    """
    gru_network, l_in, l_mask, l_gru_forward, l_split_cnn = model.bulit_gru(input_var, mask_var, input_root, input_e1, input_e2)

    # mask_train_input: where "1" is pass. where "0" isn't pass.
    mask_train_input = kbp_data.mask_train_input(training_label, num_labels=model.num_labels)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(gru_network)
    l_gru = lasagne.layers.get_output(l_gru_forward)
    l_split = lasagne.layers.get_output(l_split_cnn)

    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
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
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = T.mean(test_loss, dtype=theano.config.floatX)

    # As a bonus, also create an expression for the classification accuracy:
    test_predicted_classid = T.argmax(test_prediction, axis=1)
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)), dtype=theano.config.floatX)

    """
    4.
    train function 
    test function
    """
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var, mask_var, learning_rate_var, adam_beta1_var, input_root, input_e1, input_e2], [loss, train_acc, l_split, prediction_batch], updates=updates, on_unused_input='warn')

    # Compile a second function computing the validation loss and accuracy and F1-score:
    val_fn = theano.function(
        [input_var, target_var, mask_var, input_root, input_e1, input_e2,
         epoch_att], [test_loss, test_acc, test_predicted_classid], on_unused_input='warn')

    """
    5.start train
    """
    # Initial training variables for temporal ensembling.

    if model.network_type == 'tempens':
        ensemble_prediction = np.zeros((len(training_word_pos_vec3D), model.num_classes))
        training_targets = np.zeros((len(training_word_pos_vec3D), model.num_classes))

    scaled_unsup_weight_max = model.scaled_unsup_weight_max
    if model.num_labels != 'all':
        scaled_unsup_weight_max *= 1.0 * model.num_labels / training_label.shape[0]

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
    f1_max2 = 0
    # Max num_epoch in F1-SCORE
    f1_max_num_epochs = 0

    # Train loss
    train_loss_listplt = []
    # Test loss
    test_loss_listplt = []
    mat_conf_list = []

    aa_l_split_epoch = []

    #learning rate
    lr_list = []

    # We iterate over epochs:
    for epoch in range(model.num_epochs):

        epoch = np.int32(epoch)

        # Evaluate up/down ramps.
        rampup_value = model.rampup(epoch)
        rampdown_value = model.rampdown(epoch)

        learning_rate = rampup_value * rampdown_value * model.learning_rate_max
        lr_list.append(learning_rate)
        adam_beta1 = rampdown_value * model.adam_beta1 + (1.0 - rampdown_value) * model.rampdown_beta1_target

        # unsup_weight_var
        unsup_weight = rampup_value * scaled_unsup_weight_max
        if epoch == 0:
            unsup_weight = 0.0

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_pi_loss = 0
        train_batches = 0
        start_time = time.time()

        for batch in kbp_data.iterate_minibatches_inputAttRootE1E2(training_word_pos_vec3D,
                                                                   training_label_1hot, training_sen_length,
                                                                   model.batch_size, train_root_embedding,
                                                                   train_e1_embedding, train_e2_embedding,
                                                                   shuffle=True):
            aa_inputs, targets, mask_sen_length, input_root, input_e1, input_e2 = batch
            aa_mark_input = model.mask(mask_sen_length, model.batch_size)
            err, acc, aa_l_split, predict_each_batch = \
                train_fn(aa_inputs, targets, aa_mark_input, learning_rate, adam_beta1, input_root, input_e1, input_e2)

            train_err += err
            train_acc += acc
            train_batches += 1

        aa_l_split_epoch.append(aa_l_split)

        # train accuracy
        train_acc_listplt.append(train_acc / train_batches * 100)
        # train loss
        train_loss_listplt.append(train_err / train_batches)

        # Each epoch training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        test_predicted_classi_all = np.array([0])
        for batch in kbp_data.iterate_minibatches_inputAttRootE1E2(testing_word_pos_vec3D, \
                                                                   testing_label_1hot, testing_sen_length,
                                                                   model.batch_size, test_root_embedding,
                                                                   test_e1_embedding, test_e2_embedding, shuffle=False):
            inputs, targets, mask_sen_length, input_root, input_e1, input_e2 = batch
            mark_input = model.mask(mask_sen_length, model.batch_size)
            err, acc, test_predicted_classid = val_fn(inputs, targets, mark_input, input_root, input_e1, input_e2, epoch)

            test_err += err
            test_acc += acc
            test_batches += 1

            test_predicted_classi_all = np.concatenate((test_predicted_classi_all, test_predicted_classid))

        test_predicted_classid = np.array(test_predicted_classi_all[1:9575])

        # test accuracy
        test_acc_listplt.append(test_acc / test_batches * 100)
        # test loss
        test_loss_listplt.append(test_err / test_batches)

        # confusion matrix
        test_con_mat = confusion_matrix(testing_label[:9574], test_predicted_classid)

        # computer F1_Score
        precision, recall, f1 = model.precision_recall_f1(test_con_mat)

        f1_listplt.append(f1 * 100)
        mat_conf_list.append(test_con_mat)

        # max f1-score
        if (f1_max < f1):
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
        if (epoch % 10 == 0 and epoch != 0):
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
            # save lr picture
            model.save_plt(num_epochs, lr_list, 'learnging rate', "Learnging rate", "learnging rate", model.save_lr_path,
                           showflag=False)

    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    print("  test precision:\t\t{:.2f} %".format(precision * 100))
    print("  test recall:\t\t{:.2f} %".format(recall * 100))
    print("  test f1:\t\t{:.2f} %".format(f1 * 100))
    print("  max test f1:\t\t{:.2f} %".format(f1_max * 100))
    print("  f1 max num epochs:" + str(f1_max_num_epochs))

    #    #save the "testing_label" and "test_predicted_classid"
    #    ture_predicted_save=np.concatenate((np.reshape(test_predicted_classid,(len(test_predicted_classid),1)),np.reshape(testing_label,(len(testing_label),1))),axis=1)
    #    ture_pre_save_txt=open("../testResult/ture_predicted_save.txt","w")
    #    for i in range(len(ture_predicted_save)):
    #        ture_pre_save_txt.write(str(ture_predicted_save[i][0])+" "+str(ture_predicted_save[i][1]))
    #        ture_pre_save_txt.write("\n")
    #    ture_pre_save_txt.close()

    """
    9.save result
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

    result_file = open(model.save_result, "w")
    result_file.write("num_steps=" + str(model.num_steps) + "\n")
    result_file.write("num_epochs=" + str(model.num_epochs) + "\n")
    result_file.write("num_classes=" + str(model.num_classes) + "\n")
    result_file.write("cnn_gru_size=" + str(model.cnn_gru_size) + "\n")
    result_file.write("gru_size=" + str(model.gru_size) + "\n")
    result_file.write("keep_prob_input=" + str(model.keep_prob_input) + "\n")
    result_file.write("keep_prob_gru_output=" + str(model.keep_prob_gru_output) + "\n")
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

    """
    10.send result to my email
    """
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = msg_from
    msg['To'] = msg_to

    # Text Content
    msg.attach(MIMEText(email_content, 'plain', 'utf-8'))


    # Construct attachment and send the picture file in the current directory
    model.send_pic(model.save_picAcc_path, msg, "train_and_test_accuracy.png")
    model.send_pic(model.save_picF1_path, msg, "f1-score.png")
    model.send_pic(model.save_lr_path, msg, "lr.png")
    model.send_pic(model.save_lossTrain_path, msg, "loss_train.png")
    model.send_pic(model.save_lossTest_path, msg, "loss_test.png")

    try:
        s = smtplib.SMTP_SSL("smtp.qq.com", 465)
        s.login(msg_from, passwd)
        s.sendmail(msg_from, msg_to, msg.as_string())
        print "发送成功"
    except s.SMTPException, e:
        print "发送失败"
    finally:
=======
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2020.3.20

@author: llq
@function:
    1. realise the cross entropy in BiGRU
    2. multi-head attention
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
import sys
sys.path.append("..")

from KbpNetwork.kbpProcessData import kbpProcess
from customLayers import Input_keywords_DotLayer
from MultiheadAttention_Op import multihead_attention, multihead_attention_origin

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
        self.num_epochs = 64
        # the number of class
        self.num_classes = 40
        # the number of GRU units?
        self.cnn_gru_size = 200  # use in cnn
        self.gru_size = 300  # use in gru
        # dropout probability
        self.keep_prob_input = 0.3  # use in input
        self.keep_prob_gru_output = 0.3  # use in gru output
        # the number of entity pairs of each batch during training or testing
        self.batch_size = 100
        # learning rate
        self.learning_rate = 0.002
        # input shape
        self.embedding_len = 340
        self.input_shape = (None, 82, self.embedding_len)
        self.input_shape_att = (None, self.embedding_len)
        # mask shape
        self.mask_shape = (None, 82)
        # All gradients above this will be clipped
        self.grad_clip = 5
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
        self.rampup_length = 4
        # Ramp learning rate and Adam beta1 down during last n epochs.
        self.rampdown_length = 30
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
        # save ACCURACY picture path
        self.save_picAcc_path = "../result/train-test-accuracy.png"
        # save F1 picture path
        self.save_picF1_path = "../result/f1.png"
        self.save_picAllAcc_path = "../result/test_all_accuracy.png"
        self.save_picAllRec_path = "../result/test_all_recall.png"
        # save train loss picture path
        self.save_lossTrain_path = "../result/train-loss.png"
        # save test loss picture path
        self.save_lossTest_path = "../result/test-loss.png"
        # save lr path
        self.save_lr_path = "../result/lr.png"
        # save result file
        self.save_result = "../result/result.txt"

        """Negtive loss"""
        self.alpha = 0.5
        self.cl_lambda = 1

        """Self Attention"""
        self.attention_size2 = 2

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

        """Input attention entity and root"""
        # l_in_root = lasagne.layers.InputLayer(shape=self.input_shape_att, input_var=input_root, name="l_in_root")
        # l_in_e1 = lasagne.layers.InputLayer(shape=self.input_shape_att, input_var=input_e1, name="l_in_e1")
        # l_in_e2 = lasagne.layers.InputLayer(shape=self.input_shape_att, input_var=input_e2, name="l_in_e2")
        #
        # # Dot the "input" and "keywords"
        # l_in_root = lasagne.layers.ReshapeLayer(l_in_root, (-1, 1, self.input_shape_att[1]))
        # l_in_e1 = lasagne.layers.ReshapeLayer(l_in_e1, (-1, 1, self.input_shape_att[1]))
        # l_in_e2 = lasagne.layers.ReshapeLayer(l_in_e2, (-1, 1, self.input_shape_att[1]))
        # l_keywords = lasagne.layers.ConcatLayer([l_in_e1, l_in_root, l_in_e2], axis=1, name="l_keywords")

        """main bi-gru"""
        # Two GRU forward
        l_gru_forward = lasagne.layers.GRULayer( \
            l_input_drop, num_units=self.gru_size, grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters, updategate=self.gate_parameters,
            hidden_update=self.cell_parameters, learn_init=True,
            only_return_final=False, name="l_gru_forward1")

        l_gru_forward = lasagne.layers.GRULayer( \
            l_gru_forward, num_units=self.gru_size, grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters, updategate=self.gate_parameters,
            hidden_update=self.cell_parameters, learn_init=True,
            only_return_final=False, name="l_gru_forward2")

        # Two GRU backward
        l_gru_backward = lasagne.layers.GRULayer( \
            l_input_drop, num_units=self.gru_size, grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters, updategate=self.gate_parameters,
            hidden_update=self.cell_parameters, learn_init=True,
            only_return_final=False, backwards=True, name="l_gru_backward1")

        l_gru_backward = lasagne.layers.GRULayer( \
            l_gru_backward, num_units=self.gru_size, grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters, updategate=self.gate_parameters,
            hidden_update=self.cell_parameters, learn_init=True,
            only_return_final=False, backwards=True, name="l_gru_backward2")

        # Merge forward layers and backward layers
        l_merge = lasagne.layers.ElemwiseSumLayer([l_gru_forward, l_gru_backward])

        """auxiliary bi-gru"""
        # l_keywords = lasagne.layers.DropoutLayer(l_keywords, p=self.keep_prob_input)
        #
        # l_aux_gru_forward = lasagne.layers.GRULayer( \
        #     l_keywords, num_units=self.gru_size, grad_clipping=self.grad_clip,
        #     resetgate=self.gate_parameters, updategate=self.gate_parameters,
        #     hidden_update=self.cell_parameters, learn_init=True,
        #     only_return_final=True, name="l_aux_gru_forward")
        #
        # l_aux_gru_backward = lasagne.layers.GRULayer( \
        #     l_keywords, num_units=self.gru_size, grad_clipping=self.grad_clip,
        #     resetgate=self.gate_parameters, updategate=self.gate_parameters,
        #     hidden_update=self.cell_parameters, learn_init=True,
        #     only_return_final=True, backwards=True, name="l_aux_gru_backward")
        #
        # # Merge forward layers and backward layers
        # l_aux_merge = lasagne.layers.ElemwiseSumLayer([l_aux_gru_forward, l_aux_gru_backward])

        """multi-head attention"""
        # l_att = multihead_attention([l_keywords, l_in, l_merge, l_aux_merge], name="multihead_attention")

        for i in range(4):
            l_merge = multihead_attention_origin([l_merge, l_merge, l_merge], name="multihead_attention")

        # output dropout
        l_merge_drop = lasagne.layers.DropoutLayer(l_merge, p=self.keep_prob_gru_output)

        l_merge_fc = lasagne.layers.DenseLayer(
            l_merge_drop,
            num_units=250,
            W=lasagne.init.GlorotUniform(gain=1.0), b=lasagne.init.Constant(1.),
            nonlinearity=lasagne.nonlinearities.selu)

        l_out_margin = lasagne.layers.DenseLayer(
            l_merge_fc,
            num_units=self.num_classes,
            nonlinearity=lasagne.nonlinearities.softmax)

        return l_out_margin, l_in, l_mask, l_merge, l_merge

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

    def send_pic(self, pic_path, msg, pic_name):
        # Construct attachment 1 and send the picture file in the current directory
        att1 = MIMEText(open(pic_path, 'rb').read(), 'base64', 'utf-8')
        att1["Content-Type"] = 'application/octet-stream'
        # filename in email display
        att1["Content-Disposition"] = 'attachment; filename="' + pic_name + '"'
        msg.attach(att1)


if __name__ == "__main__":

    """
    1.Loading data:training data and test data
    """
    print("Loading data")
    # 1.Class:Process_data() and init the dict_word_vec
    kbp_data = kbpProcess()
    kbp_data.dict_word2vec()
    kbp_data.label2id_init()

    # 2.traing_word_pos_vec3D:training data
    training_word_pos_vec3D, training_sen_length, train_sen_list2D = \
        kbp_data.embedding_lookup(kbp_data.train_sen_store_filename, \
                                  kbp_data.training_e1_e2_pos_filename, kbp_data.training_sen_number)
    training_word_pos_vec3D = np.float32(training_word_pos_vec3D)
    training_sen_length = np.int32(np.array(training_sen_length))

    # 3.testing_word_pos_vec3D:testing data
    testing_word_pos_vec3D, testing_sen_length, test_sen_list2D = kbp_data.embedding_lookup(
        kbp_data.test_sen_store_filename, \
        kbp_data.testing_e1_e2_pos_filename, kbp_data.testing_sen_number)
    testing_word_pos_vec3D = np.float32(testing_word_pos_vec3D)
    testing_sen_length = np.int32(np.array(testing_sen_length))

    # 4.training label:8000
    #  mask_train:unsupervised
    training_label = kbp_data.label2id_in_data(kbp_data.train_label_store_filename, \
                                               kbp_data.training_sen_number)
    training_label = np.int32(training_label)

    # 5.testing label:10825
    testing_label = kbp_data.label2id_in_data(kbp_data.test_label_store_filename, \
                                              kbp_data.testing_sen_number)
    testing_label = np.int32(testing_label)

    # 6.label id value: Change the label to id.And 10 classes number(0-9)
    label2id = kbp_data.label2id

    # 7.One-hot encode
    training_label_1hot = kbp_data.label2id_1hot(training_label, label2id)
    training_label_1hot = np.int32(training_label_1hot)

    testing_label_1hot = kbp_data.label2id_1hot(testing_label, label2id)
    testing_label_1hot = np.int32(testing_label_1hot)

    # 8.embedding root,e1 and e2
    train_root_embedding, train_e1_embedding, train_e2_embedding = \
        kbp_data.embedding_looking_root_e1_e2(kbp_data.e1_sdp_train_file, kbp_data.e2_sdp_train_file,
                                              kbp_data.training_sen_number, train_sen_list2D)

    test_root_embedding, test_e1_embedding, test_e2_embedding = \
        kbp_data.embedding_looking_root_e1_e2(kbp_data.e1_sdp_test_file, kbp_data.e2_sdp_test_file,
                                              kbp_data.testing_sen_number, test_sen_list2D)

    # 9.mask_train_input:unsupervised.
    # Get in pro_data.mask_train_input()
    # Random shuffle.
    indices = np.arange(len(training_word_pos_vec3D))
    np.random.shuffle(indices)
    training_word_pos_vec3D = training_word_pos_vec3D[indices]
    training_sen_length = training_sen_length[indices]
    training_label_1hot = training_label_1hot[indices]
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

    # input attention entity and root
    input_root = T.fmatrix("input_root")
    input_e1 = T.fmatrix("input_e1")
    input_e2 = T.fmatrix("input_e2")
    epoch_att = T.iscalar("epoch_att")

    # loss parameter
    centers = theano.shared(np.float32(np.zeros([model.num_classes, 250])), "centers")

    """
    2.
    Bulit GRU network
    ADAM
    """
    gru_network, l_in, l_mask, l_gru_forward, l_split_cnn = model.bulit_gru(input_var, mask_var, input_root, input_e1, input_e2)

    # mask_train_input: where "1" is pass. where "0" isn't pass.
    mask_train_input = kbp_data.mask_train_input(training_label, num_labels=model.num_labels)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(gru_network)
    l_gru = lasagne.layers.get_output(l_gru_forward)
    l_split = lasagne.layers.get_output(l_split_cnn)

    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
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
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = T.mean(test_loss, dtype=theano.config.floatX)

    # As a bonus, also create an expression for the classification accuracy:
    test_predicted_classid = T.argmax(test_prediction, axis=1)
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)), dtype=theano.config.floatX)

    """
    4.
    train function 
    test function
    """
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var, mask_var, learning_rate_var, adam_beta1_var, input_root, input_e1, input_e2], [loss, train_acc, l_split, prediction_batch], updates=updates, on_unused_input='warn')

    # Compile a second function computing the validation loss and accuracy and F1-score:
    val_fn = theano.function(
        [input_var, target_var, mask_var, input_root, input_e1, input_e2,
         epoch_att], [test_loss, test_acc, test_predicted_classid], on_unused_input='warn')

    """
    5.start train
    """
    # Initial training variables for temporal ensembling.

    if model.network_type == 'tempens':
        ensemble_prediction = np.zeros((len(training_word_pos_vec3D), model.num_classes))
        training_targets = np.zeros((len(training_word_pos_vec3D), model.num_classes))

    scaled_unsup_weight_max = model.scaled_unsup_weight_max
    if model.num_labels != 'all':
        scaled_unsup_weight_max *= 1.0 * model.num_labels / training_label.shape[0]

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
    f1_max2 = 0
    # Max num_epoch in F1-SCORE
    f1_max_num_epochs = 0

    # Train loss
    train_loss_listplt = []
    # Test loss
    test_loss_listplt = []
    mat_conf_list = []

    aa_l_split_epoch = []

    #learning rate
    lr_list = []

    # We iterate over epochs:
    for epoch in range(model.num_epochs):

        epoch = np.int32(epoch)

        # Evaluate up/down ramps.
        rampup_value = model.rampup(epoch)
        rampdown_value = model.rampdown(epoch)

        learning_rate = rampup_value * rampdown_value * model.learning_rate_max
        lr_list.append(learning_rate)
        adam_beta1 = rampdown_value * model.adam_beta1 + (1.0 - rampdown_value) * model.rampdown_beta1_target

        # unsup_weight_var
        unsup_weight = rampup_value * scaled_unsup_weight_max
        if epoch == 0:
            unsup_weight = 0.0

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_pi_loss = 0
        train_batches = 0
        start_time = time.time()

        for batch in kbp_data.iterate_minibatches_inputAttRootE1E2(training_word_pos_vec3D,
                                                                   training_label_1hot, training_sen_length,
                                                                   model.batch_size, train_root_embedding,
                                                                   train_e1_embedding, train_e2_embedding,
                                                                   shuffle=True):
            aa_inputs, targets, mask_sen_length, input_root, input_e1, input_e2 = batch
            aa_mark_input = model.mask(mask_sen_length, model.batch_size)
            err, acc, aa_l_split, predict_each_batch = \
                train_fn(aa_inputs, targets, aa_mark_input, learning_rate, adam_beta1, input_root, input_e1, input_e2)

            train_err += err
            train_acc += acc
            train_batches += 1

        aa_l_split_epoch.append(aa_l_split)

        # train accuracy
        train_acc_listplt.append(train_acc / train_batches * 100)
        # train loss
        train_loss_listplt.append(train_err / train_batches)

        # Each epoch training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        test_predicted_classi_all = np.array([0])
        for batch in kbp_data.iterate_minibatches_inputAttRootE1E2(testing_word_pos_vec3D, \
                                                                   testing_label_1hot, testing_sen_length,
                                                                   model.batch_size, test_root_embedding,
                                                                   test_e1_embedding, test_e2_embedding, shuffle=False):
            inputs, targets, mask_sen_length, input_root, input_e1, input_e2 = batch
            mark_input = model.mask(mask_sen_length, model.batch_size)
            err, acc, test_predicted_classid = val_fn(inputs, targets, mark_input, input_root, input_e1, input_e2, epoch)

            test_err += err
            test_acc += acc
            test_batches += 1

            test_predicted_classi_all = np.concatenate((test_predicted_classi_all, test_predicted_classid))

        test_predicted_classid = np.array(test_predicted_classi_all[1:9575])

        # test accuracy
        test_acc_listplt.append(test_acc / test_batches * 100)
        # test loss
        test_loss_listplt.append(test_err / test_batches)

        # confusion matrix
        test_con_mat = confusion_matrix(testing_label[:9574], test_predicted_classid)

        # computer F1_Score
        precision, recall, f1 = model.precision_recall_f1(test_con_mat)

        f1_listplt.append(f1 * 100)
        mat_conf_list.append(test_con_mat)

        # max f1-score
        if (f1_max < f1):
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
        if (epoch % 10 == 0 and epoch != 0):
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
            # save lr picture
            model.save_plt(num_epochs, lr_list, 'learnging rate', "Learnging rate", "learnging rate", model.save_lr_path,
                           showflag=False)

    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    print("  test precision:\t\t{:.2f} %".format(precision * 100))
    print("  test recall:\t\t{:.2f} %".format(recall * 100))
    print("  test f1:\t\t{:.2f} %".format(f1 * 100))
    print("  max test f1:\t\t{:.2f} %".format(f1_max * 100))
    print("  f1 max num epochs:" + str(f1_max_num_epochs))

    #    #save the "testing_label" and "test_predicted_classid"
    #    ture_predicted_save=np.concatenate((np.reshape(test_predicted_classid,(len(test_predicted_classid),1)),np.reshape(testing_label,(len(testing_label),1))),axis=1)
    #    ture_pre_save_txt=open("../testResult/ture_predicted_save.txt","w")
    #    for i in range(len(ture_predicted_save)):
    #        ture_pre_save_txt.write(str(ture_predicted_save[i][0])+" "+str(ture_predicted_save[i][1]))
    #        ture_pre_save_txt.write("\n")
    #    ture_pre_save_txt.close()

    """
    9.save result
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

    result_file = open(model.save_result, "w")
    result_file.write("num_steps=" + str(model.num_steps) + "\n")
    result_file.write("num_epochs=" + str(model.num_epochs) + "\n")
    result_file.write("num_classes=" + str(model.num_classes) + "\n")
    result_file.write("cnn_gru_size=" + str(model.cnn_gru_size) + "\n")
    result_file.write("gru_size=" + str(model.gru_size) + "\n")
    result_file.write("keep_prob_input=" + str(model.keep_prob_input) + "\n")
    result_file.write("keep_prob_gru_output=" + str(model.keep_prob_gru_output) + "\n")
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

    """
    10.send result to my email
    """
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = msg_from
    msg['To'] = msg_to

    # Text Content
    msg.attach(MIMEText(email_content, 'plain', 'utf-8'))


    # Construct attachment and send the picture file in the current directory
    model.send_pic(model.save_picAcc_path, msg, "train_and_test_accuracy.png")
    model.send_pic(model.save_picF1_path, msg, "f1-score.png")
    model.send_pic(model.save_lr_path, msg, "lr.png")
    model.send_pic(model.save_lossTrain_path, msg, "loss_train.png")
    model.send_pic(model.save_lossTest_path, msg, "loss_test.png")

    try:
        s = smtplib.SMTP_SSL("smtp.qq.com", 465)
        s.login(msg_from, passwd)
        s.sendmail(msg_from, msg_to, msg.as_string())
        print "发送成功"
    except s.SMTPException, e:
        print "发送失败"
    finally:
>>>>>>> 9e9c62ef716e71fd5ce04a1694098a1af0372f05
        s.quit()