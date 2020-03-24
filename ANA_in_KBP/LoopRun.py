#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2020.3.22

@author: llq
@function:
    1. loop call the function
"""
import sys
sys.path.append("..")
from ANA_in_KBP.ANACeBiGRU import Network
import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import confusion_matrix


from KbpNetwork.kbpProcessData import kbpProcess

theano.config.floatX = "float32"

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

msg_from = '1182953475@qq.com'
passwd = 'kicjjxxrunufjfej'
msg_to = '1182953475@qq.com'
subject = "Network test1"  # Theme


def loop_run(model):
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
        s.quit()


if __name__ == "__main__":

    for _ in range(3):
        model = Network()
        loop_run(model)