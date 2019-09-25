#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:10:02 2018

Negtive loss+cnn_gru

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
from sklearn.metrics import precision_recall_fscore_support,f1_score,classification_report,confusion_matrix

from sdpProcessData import sdpProcessData
from CustomLayers import SplitInLeft,SplitInRight,SplitInGlobal,HighwayNetwork1D,HighwayNetwork2D,MarginLossLayer
from CustomLoss import Negative_loss,Margin_loss

theano.config.floatX="float32"

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
msg_from='1182953475@qq.com'                                 
passwd='kicjjxxrunufjfej'                                   
msg_to='1182953475@qq.com'   
subject="Network test1"     #Theme

class Network:

    def __init__(self):
        """network parameters"""
        #the number of unrolled steps of LSTM
        self.num_steps = 31
        #the number of epoch(one epoch=N iterations)
        self.num_epochs = 100
        #the number of class
        self.num_classes = 19
        #the number of GRU units?
        self.cnn_gru_size = 200  #use in cnn
        self.gru_size = 150      #use in gru
        #dropout probability
        self.keep_prob_input=0.5                #use in input
        self.keep_prob_gru_output=0.5           #use in gru
        self.keep_prob_cnn=0.5                  #use in cnn
        self.keep_prob_cnn_gur_output=0.5       #use in output
        # the number of entity pairs of each batch during training or testing
        self.batch_size = 100
        #learning rate
        self.learning_rate=0.001
        #input shape
        self.input_shape=(None,None,360)
        #mask shape
        self.mask_shape=(None,31)
        # All gradients above this will be clipped
        self.grad_clip = 5
        #l2_loss
        self.l2_loss=1e-4
        # Choose "pi" or "tempens" or "ordinary"
        self.network_type="ordinary"
        
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
        self.rampup_length=30
        # Ramp learning rate and Adam beta1 down during last n epochs.
        self.rampdown_length=50
        # Unsupervised loss maximum (w_max in paper). Set to 0.0 -> supervised loss only.
        self.scaled_unsup_weight_max=100.0
        # Maximum learning rate.
        self.learning_rate_max=0.001
        # Default value.
        self.adam_beta1= 0.9
        # Target value for Adam beta1 for rampdown.
        self.rampdown_beta1_target= 0.5                     
        # Total number of labeled inputs (1/10th of this per class). Value 'all' uses all labels.
        self.num_labels='all'
        # Ensemble prediction decay constant (\alpha in paper).
        self.prediction_decay= 0.6    
        
        """Save Picture"""
        #save ACCURACY picture path
        self.save_picAcc_path="../result/train-test-accuracy.jpg"
        #save F1 picture path
        self.save_picF1_path="../result/f1-score.jpg"
        #save train loss picture path
        self.save_lossTrain_path="../result/train-loss.jpg"
        #save test loss picture path
        self.save_lossTest_path="../result/test-loss.jpg"
        #save result file
        self.save_result="../result/result.txt"
        
        """Negtive loss"""
        self.negative_loss_alpha=np.float32(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
        self.negative_loss_lamda=1
        
    def bulit_gru(self,input_var=None,mask_var=None, left_sdp_length=None, sen_length=None):
        """
        Bulit the GRU network
        """
        #inputlayer
        l_in=lasagne.layers.InputLayer(shape=self.input_shape,input_var=input_var,name="l_in")
       
        #mask layer
        l_mask=lasagne.layers.InputLayer(shape=self.mask_shape,input_var=mask_var,name="l_mask")
   
        #inpute dropout
        l_input_drop=lasagne.layers.DropoutLayer(l_in,p=self.keep_prob_input)
        '''
        """
        CNN
        """
        #the length of sentences
        l_sen_length=lasagne.layers.InputLayer(shape=(None,1),input_var=sen_length,name="l_sen_length")

        #split the global SDP
        l_split_global_sdp=SplitInGlobal((l_sen_length,l_in))
        l_split_global_sdp=lasagne.layers.ReshapeLayer(l_split_global_sdp,([0],1,[1],[2]))

        l_global_sdp_cnn = lasagne.layers.Conv2DLayer(
            l_split_global_sdp, num_filters=500, filter_size=(3, self.cnn_gru_size),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        
        # MAX-pooling in global SDP
        l_global_sdp_maxpooling=lasagne.layers.GlobalPoolLayer(l_global_sdp_cnn, pool_function=T.max)
        
        #output dropout
        l_sdp_drop=lasagne.layers.DropoutLayer(l_global_sdp_maxpooling,p=self.keep_prob_cnn)

        w_classes_init=np.sqrt(6.0/(19+500))
        l_out_margin=MarginLossLayer(l_sdp_drop,w_classes=lasagne.init.Uniform(w_classes_init), class_number=19)
        
        return l_out_margin,l_in,l_mask,l_global_sdp_maxpooling,l_out_margin
        '''
        
        """
        1.Split third input.
        2.Input third CNN
        """
        #GRU forward
        l_gru_forward=lasagne.layers.GRULayer(\
            l_input_drop,num_units=self.cnn_gru_size,mask_input=l_mask,grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters,updategate=self.gate_parameters,
            hidden_update=self.cell_parameters,learn_init=True,
            only_return_final=False,name="l_gru_forward")
        
        #left sdp length input layer
        l_left_sdp_length=lasagne.layers.InputLayer(shape=(None,1),input_var=left_sdp_length,name="l_left_sdp_length")
        
        #the length of sentences
        l_sen_length=lasagne.layers.InputLayer(shape=(None,1),input_var=sen_length,name="l_sen_length")
        
        #split the left SDP and the right SDP
        l_split_left_sdp=SplitInLeft((l_left_sdp_length,l_gru_forward))
        l_split_right_sdp=SplitInRight((l_left_sdp_length,l_sen_length,l_gru_forward))
        #split the global SDP
        l_split_global_sdp=SplitInGlobal((l_sen_length,l_gru_forward))
        
        #Reshape the layer in 4D
        l_split_left_sdp=lasagne.layers.ReshapeLayer(l_split_left_sdp,([0],1,[1],[2]))
        l_split_right_sdp=lasagne.layers.ReshapeLayer(l_split_right_sdp,([0],1,[1],[2]))
        l_split_global_sdp=lasagne.layers.ReshapeLayer(l_split_global_sdp,([0],1,[1],[2]))

        # Convolutional layer in left SDP     
        l_left_sdp_cnn = lasagne.layers.Conv2DLayer(
            l_split_left_sdp, num_filters=50, filter_size=(2, self.cnn_gru_size),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        
        # Convolutional layer in right SDP
        l_right_sdp_cnn = lasagne.layers.Conv2DLayer(
            l_split_right_sdp, num_filters=50, filter_size=(2, self.cnn_gru_size),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        
        # Convolutional layer in global SDP
        l_global_sdp_cnn = lasagne.layers.Conv2DLayer(
            l_split_global_sdp, num_filters=50, filter_size=(2, self.cnn_gru_size),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        
        # Max-pooling in left SDP
        l_left_sdp_maxpooling=lasagne.layers.ReshapeLayer(
            lasagne.layers.GlobalPoolLayer(l_left_sdp_cnn, pool_function=T.max),
            ([0],[1],1))
        
        # MAX-pooling in rigth SDP
        l_right_sdp_maxpooling=lasagne.layers.ReshapeLayer(
            lasagne.layers.GlobalPoolLayer(l_right_sdp_cnn, pool_function=T.max),
            ([0],[1],1))
        
        # MAX-pooling in global SDP
        l_global_sdp_maxpooling=lasagne.layers.ReshapeLayer(
            lasagne.layers.GlobalPoolLayer(l_global_sdp_cnn, pool_function=T.max),
            ([0],[1],1))
        
        #Concatenate left SDP and right SDP
        l_con_sdp=lasagne.layers.ConcatLayer((l_global_sdp_maxpooling,l_left_sdp_maxpooling,l_right_sdp_maxpooling),axis=2)
       
        #output dropout
        l_con_sdp_drop=lasagne.layers.DropoutLayer(l_con_sdp,p=self.keep_prob_cnn)
        
        # A fully-connected layer of 200 units with 50% dropout on its inputs:
        l_con_sdp_den = lasagne.layers.DenseLayer(
            l_con_sdp_drop,
            num_units=150,
            nonlinearity=lasagne.nonlinearities.rectify)

#        # Finally, we'll add the fully-connected output layer, of 10 softmax units:
#        l_out = lasagne.layers.DenseLayer(\
#            lasagne.layers.dropout(l_con_sdp_den, p=self.keep_prob_backward), \
#            num_units=self.num_classes,\
#            nonlinearity=lasagne.nonlinearities.softmax)
        
        """
        2.Input in GRU
        """
        l_gru_forward2=lasagne.layers.GRULayer(\
            l_input_drop,num_units=self.gru_size,mask_input=l_mask,grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters,updategate=self.gate_parameters,
            hidden_update=self.cell_parameters,learn_init=True,
            only_return_final=True,name="l_gru_forward2")
        
        #GRU backward
        l_gru_backward=lasagne.layers.GRULayer(\
            l_input_drop,num_units=self.gru_size,mask_input=l_mask,grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters,updategate=self.gate_parameters,
            hidden_update=self.cell_parameters,learn_init=True,
            only_return_final=True,backwards=True,name="l_gru_backward")
                      
        #Merge forward layers and backward layers
        l_merge=lasagne.layers.ElemwiseSumLayer([l_gru_forward2,l_gru_backward])
        
        #output dropout
        l_merge_drop=lasagne.layers.DropoutLayer(l_merge,p=self.keep_prob_gru_output)
        
        """
        3.Concatenate CNN and GRU
        """
        l_cnn_gru=lasagne.layers.ConcatLayer((l_con_sdp_den,l_merge_drop),axis=1)
        
#        #Highway network
#        l_highway_network=HighwayNetwork1D(l_cnn_gru)
        
        #Margin loss
        #init w_classes: it's format is (class_number,network_output[1])
        w_classes_init=np.sqrt(6.0/(19+self.gru_size+150))
        l_out_margin=MarginLossLayer(l_cnn_gru,w_classes=lasagne.init.Uniform(w_classes_init), class_number=19)
        a=lasagne.layers.ReshapeLayer(
          lasagne.layers.InputLayer(shape=(19,300),input_var=l_out_margin.w_classes),
          (19,300))
        
#        l_out = lasagne.layers.DenseLayer(\
#            lasagne.layers.dropout(l_cnn_gru, p=self.keep_prob_cnn_gur_output),\
#            num_units=self.num_classes,\
#            nonlinearity=lasagne.nonlinearities.softmax)
        
        return l_out_margin,l_in,l_mask,a,l_out_margin
        
    def mask(self,mask_var,batch_size):
        """
        mask input:it's size is (batch_size,100)
        When mask[i,j]=1,so it can input.
        When mask[i,j]=0,so it can't input.
        """
        mark_input=np.int32(np.zeros((batch_size,self.num_steps)))
        sentence=0
        for length in mask_var:
            mark_input[sentence]=np.concatenate((np.ones((1,length)),np.zeros((1,self.num_steps-length))),axis=1)
            sentence=sentence+1
        return mark_input
    
    def save_plt(self,x,y,label,title,ylabel_name,save_path,twice=False,x2=None,y2=None,label2=None,showflag=True):
        """
        Save picture:
            1.Train Accuracy
            2.Test Accuracy
            3.Test F1-Score
            4.Train loss
            5.Test loss
        """
        plt.figure(figsize=(10,10))
        plt.title(title)
        plt.grid() #open grid
        plt.xlabel('number epoch')
        plt.ylabel(ylabel_name)
        #print two curve
        if twice:
            plt.plot(x,y,label=label)
            plt.legend(loc='best')
            plt.plot(x2,y2,label=label2)
            plt.legend(loc='best')
        else:
            plt.plot(x,y,label=label)
            plt.legend(loc='best')

        plt.savefig(save_path)
        if showflag:
            plt.show()
        plt.close()
    
    def accuracy_f1(self,con_mat,support):
        """
        Compute the Accuracy and Recall and F1-SCORE with confusion_matrix
        """
        accuracy=0.0
        recall=0.0
        for i in range(0,18,2):
            try:
                accuracy+=float(con_mat[i][i]+con_mat[i+1][i+1])/float(np.sum(con_mat[:,i])+np.sum(con_mat[:,i+1]))
                recall+=float(con_mat[i][i]+con_mat[i+1][i+1])/float(support[i]+support[i+1])
            except:
                print "ZeroDivisionError: float division by zero"
                return 0  
                
        accuracy=accuracy/9.0
        recall=recall/9.0
        f1=(2*accuracy*recall)/(accuracy+recall)
        return f1
    
    def rampup(self,epoch):
        """
        Training utils.
        """
        if epoch < self.rampup_length:
            p = max(0.0, float(epoch)) / float(self.rampup_length)
            p = 1.0 - p
            return math.exp(-p*p*5.0)
        else:
            return 1.0
    
    def rampdown(self,epoch):
        if epoch >= (self.num_epochs - self.rampdown_length):
            ep = (epoch - (self.num_epochs - self.rampdown_length)) * 0.5
            return math.exp(-(ep * ep) / self.rampdown_length)
        else:
            return 1.0
    
if __name__=="__main__":
    
    """
    1.Loading data:training data and test data
    """
    print("Loading data")
    #1.Class:Process_data() and init the dict_word_vec
    sdp_pro=sdpProcessData()
    sdp_pro.dict_word2vec()
    sdp_pro.label2id_init()
    sdp_pro.dep_vec_init()
    
    #3(1).combine the two SDP in train data
    con_spd_train=sdp_pro.combine_sdp(sdp_pro.e1_sdp_train_file,sdp_pro.e2_sdp_train_file)
    
    #3(2).traing_word_pos_vec3D:training data
    sdp_pro.training_word_pos_vec3D,sdp_pro.training_sen_length,dep_vec=sdp_pro.embedding_lookup(con_spd_train,sdp_pro.training_word_vec3D,sdp_pro.training_sen_number,sdp_pro.train_sdp_link_dep_file)
    training_word_pos_vec3D=np.float32(sdp_pro.training_word_pos_vec3D)
    training_sen_length=np.int32(np.array(sdp_pro.training_sen_length))

    #3(3).left sdp length in traing data
    train_left_sdp_length=np.int32(np.reshape(sdp_pro.left_sdp_length(sdp_pro.train_e1_sdp_pos_file),(-1,1)))

    #4(1).combine the two SDP in test data
    con_spd_test=sdp_pro.combine_sdp(sdp_pro.e1_sdp_test_file,sdp_pro.e2_sdp_test_file)
 
    #4(2).traing_word_pos_vec3D:test data
    sdp_pro.testing_word_pos_vec3D,sdp_pro.testing_sen_length,dep_vec=sdp_pro.embedding_lookup(con_spd_test,sdp_pro.testing_word_vec3D,sdp_pro.testing_sen_number,sdp_pro.test_sdp_link_dep_file)
    testing_word_pos_vec3D=np.float32(sdp_pro.testing_word_pos_vec3D)
    testing_sen_length=np.int32(np.array(sdp_pro.testing_sen_length))

    #4(3).left sdp length in test data
    test_left_sdp_length=np.int32(np.reshape(sdp_pro.left_sdp_length(sdp_pro.test_e1_sdp_pos_file),(-1,1)))
 
    #5(1).training label:8000
    #  mask_train:unsupervised
    sdp_pro.training_label=sdp_pro.label2id_in_data(sdp_pro.train_label_store_filename,\
      sdp_pro.training_label)
    training_label=np.int32(sdp_pro.training_label)
        
    #5(2).testing label:2717
    sdp_pro.testing_label=sdp_pro.label2id_in_data(sdp_pro.test_label_store_filename,\
      sdp_pro.testing_label)
    testing_label=np.int32(sdp_pro.testing_label)    

    #6.One-hot encode
    #label id value: Change the label to id.And 10 classes number(0-9)
    label2id=sdp_pro.label2id
    training_label_1hot=sdp_pro.label2id_1hot(training_label,label2id)
    training_label_1hot=np.int32(training_label_1hot)
    
    testing_label_1hot=sdp_pro.label2id_1hot(testing_label,label2id)    
    testing_label_1hot=np.int32(testing_label_1hot)
    
    #7.mask_train_input:unsupervised.
    #Get in pro_data.mask_train_input()
    # Random shuffle.
    indices = np.arange(len(training_word_pos_vec3D))
    np.random.shuffle(indices)
    training_word_pos_vec3D = training_word_pos_vec3D[indices]
    training_sen_length = training_sen_length[indices]
    training_label=training_label[indices]
    training_label_1hot=training_label_1hot[indices]
    train_left_sdp_length=train_left_sdp_length[indices]
    
    """
    new model
    """
    model=Network()
    
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor3('inputs')
    target_var = T.imatrix('targets')
    mask_var = T.imatrix('mask_layer')
    #Pi model variables:
    if model.network_type=="pi":
        input_b_var = T.tensor3('inputs_b')
        mask_train=T.vector('mask_train')
        unsup_weight_var = T.scalar('unsup_weight')
    elif model.network_type=="tempens":
    #tempens model variables:
        z_target_var = T.matrix('z_targets')
        mask_train = T.vector('mask_train')
        unsup_weight_var = T.scalar('unsup_weight')
    
    learning_rate_var = T.scalar('learning_rate')
    adam_beta1_var = T.scalar('adam_beta1')
    
    #Left sdp length
    left_sdp_length=T.imatrix('left_sdp_length')
    #Sentences length
    sen_length=T.imatrix('sen_length')
    
    #negative loss
    negative_loss_alpha=T.fvector("negative_loss_alpha")
    negative_loss_lamda=T.fscalar("negative_loss_lamda") 
    
    """
    2.
    Bulit GRU network
    ADAM
    """
    gru_network,l_in,l_mask,l_gru_forward,l_split_cnn=model.bulit_gru(input_var,mask_var,left_sdp_length,sen_length)
    
    #mask_train_input: where "1" is pass. where "0" isn't pass.
    mask_train_input=sdp_pro.mask_train_input(training_label,num_labels=model.num_labels)
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(gru_network)
    l_gru = lasagne.layers.get_output(l_gru_forward)
    l_split = lasagne.layers.get_output(l_split_cnn)
    aaa_w_classes = l_split_cnn.w_classes.get_value()
    
#    loss = Negative_loss(prediction, target_var, negative_loss_alpha, negative_loss_lamda)
    loss = Margin_loss(prediction, target_var)

    #Pi model loss
    if model.network_type=="pi":
        loss = T.mean(loss * mask_train, dtype=theano.config.floatX) 
        train_prediction_b = lasagne.layers.get_output(gru_network, inputs={l_in:input_b_var,l_mask:mask_var}) # Second branch.
        loss += unsup_weight_var * T.mean(lasagne.objectives.squared_error(prediction, train_prediction_b))
    elif model.network_type=="tempens":
        #Tempens model loss:
        loss = T.mean(loss * mask_train, dtype=theano.config.floatX)
        loss += unsup_weight_var * T.mean(lasagne.objectives.squared_error(prediction, z_target_var))
    else:
        loss = T.mean(loss, dtype=theano.config.floatX)
        
    #regularization:L1,L2
    l2_penalty = lasagne.regularization.regularize_network_params(gru_network, lasagne.regularization.l2) * model.l2_loss
    loss=loss+l2_penalty
    
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
#    test_loss = Negative_loss(test_prediction,target_var,negative_loss_alpha,negative_loss_lamda)
    test_loss = Margin_loss(test_prediction,target_var)
    test_loss = T.mean(test_loss,dtype=theano.config.floatX)
    
    # As a bonus, also create an expression for the classification accuracy:
    #????????????????????
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
        train_fn = theano.function([input_var, target_var, mask_var, learning_rate_var, adam_beta1_var, left_sdp_length, sen_length, negative_loss_alpha, negative_loss_lamda], [loss,train_acc,l_gru,l_split], updates=updates, on_unused_input='warn')
    
    # Compile a second function computing the validation loss and accuracy and F1-score:
    val_fn = theano.function([input_var, target_var, mask_var, left_sdp_length, sen_length, negative_loss_alpha, negative_loss_lamda], [test_loss, test_acc, test_predicted_classid], on_unused_input='warn')
    
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

    email_content=""
      
    #Train accuracy list
    train_acc_listplt=[]
    #Test accuracy list
    test_acc_listplt=[]
    
    #F1-SCORE list:in picture
    f1_score_listplt=[]
    #Max F1-SCORE
    f1_score_max=0
    #Max num_epoch in F1-SCORE
    f1_max_num_epochs=0
    
    #Train loss
    train_loss_listplt=[]
    #Test loss
    test_loss_listplt=[]
    
    
    # We iterate over epochs:
    for epoch in range(model.num_epochs):
        
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
            epoch_predictions = np.zeros((len(training_word_pos_vec3D), model.num_classes))
            epoch_execmask = np.zeros(len(training_word_pos_vec3D)) # Which inputs were executed.
            training_targets = floatX(training_targets)    
        
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc=0
        train_batches = 0
        start_time = time.time()
        
        #Pi model:run batch
        if model.network_type=="pi":
            for batch in sdp_pro.iterate_minibatches_pi(training_word_pos_vec3D, \
              training_label, training_sen_length, model.batch_size, mask_train_input, shuffle=True):
                inputs, targets, mask_sen_length, mask_train = batch
                mark_input=model.mask(mask_sen_length,model.batch_size)
                err ,acc = train_fn(inputs, targets, mark_input, inputs, mask_train, unsup_weight, learning_rate, adam_beta1)
                train_err+=err
                train_acc+=acc
                train_batches += 1
        elif model.network_type=="tempens":
        #Tempens model:run batch
            for batch in sdp_pro.iterate_minibatches_tempens(training_word_pos_vec3D, \
              training_label, training_sen_length, model.batch_size, mask_train_input, training_targets, shuffle=True):
                inputs, targets, mask_sen_length, mask_train, z_targets, indices = batch
                mark_input=model.mask(mask_sen_length,model.batch_size)
                err ,acc, prediction = train_fn(inputs, targets, mark_input, z_targets, mask_train, unsup_weight, learning_rate, adam_beta1)
                for i, j in enumerate(indices):
                    epoch_predictions[j] = prediction[i] # Gather epoch predictions.
                    epoch_execmask[j] = 1.0
                train_err+=err
                train_acc+=acc
                train_batches += 1
        else:
            #Ordinary model
            for batch in sdp_pro.iterate_minibatches(training_word_pos_vec3D, \
              training_label_1hot, training_sen_length, train_left_sdp_length, model.batch_size, shuffle=True):
                aa_inputs, targets, mask_sen_length, aa_left_sdp_length = batch
                aa_mark_input=model.mask(mask_sen_length,model.batch_size)
                err ,acc, aa_l_gru,aa_l_split = train_fn(aa_inputs, targets, aa_mark_input, learning_rate, adam_beta1, aa_left_sdp_length, np.reshape(mask_sen_length,(-1,1)), model.negative_loss_alpha, model.negative_loss_lamda)
                train_err+=err
                train_acc+=acc
                train_batches += 1
                
        #train accuracy
        train_acc_listplt.append(train_acc / train_batches * 100)
        #train loss
        train_loss_listplt.append(train_err / train_batches)
        
        # Each epoch training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
#        test_predicted_classi_all=np.array([0])
        for batch in sdp_pro.iterate_minibatches(testing_word_pos_vec3D, \
          testing_label_1hot, testing_sen_length, test_left_sdp_length, model.batch_size, shuffle=False):
            inputs, targets, mask_sen_length, left_sdp_length = batch
            mark_input=model.mask(mask_sen_length,model.batch_size)
            err, acc, test_predicted_classid = val_fn(inputs, targets, mark_input, left_sdp_length, np.reshape(mask_sen_length,(-1,1)), model.negative_loss_alpha, model.negative_loss_lamda)
            test_err += err
            test_acc += acc
            test_batches += 1
            
#            test_predicted_classi_all=np.concatenate((test_predicted_classi_all,test_predicted_classid))
        
#        test_predicted_classid=np.array(test_predicted_classi_all[1:])
        #test accuracy
        test_acc_listplt.append(test_acc / test_batches * 100)
        #test loss
        test_loss_listplt.append(test_err / test_batches)
        
        #update "ensemble_prediction(Z)" and "training_targets"
        if model.network_type == 'tempens':
            # Basic mode.
            ensemble_prediction = (model.prediction_decay * ensemble_prediction) + (1.0 - model.prediction_decay) * epoch_predictions
            training_targets = ensemble_prediction / (1.0 - model.prediction_decay ** ((epoch - 0) + 1.0))

        #F1 value
        mark_input=model.mask(testing_sen_length,len(testing_sen_length))
        err, acc, test_predicted_classid=val_fn(testing_word_pos_vec3D, testing_label_1hot, mark_input, test_left_sdp_length, np.reshape(testing_sen_length,(-1,1)), model.negative_loss_alpha, model.negative_loss_lamda)
        #confusion matrix
        test_con_mat=confusion_matrix(testing_label,test_predicted_classid)
        #support
        _,_,_,support=precision_recall_fscore_support(testing_label,test_predicted_classid)   
        #computer F1_Score
        test_f1_score=model.accuracy_f1(test_con_mat,support)
        
        f1_score_listplt.append(test_f1_score*100)
        
        #max f1-score
        if(f1_score_max<test_f1_score):
            f1_score_max=test_f1_score
            f1_max_num_epochs=epoch+1
        
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, model.num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.2f} %".format(train_acc / train_batches * 100))
        #Testing loss and accuracy
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))
        print("  test f1 score:\t\t{:.2f} %".format(test_f1_score*100))
        print("  max  f1 score:\t\t{:.2f} %".format(f1_score_max*100))
        print("  f1 max num epochs:\t\t    "+str(f1_max_num_epochs))
        
        #Email content
        email_content=email_content+"Epoch {} of {} took {:.3f}s".format(
            epoch + 1, model.num_epochs, time.time() - start_time)+"\n"
        email_content=email_content+"  training loss:\t\t{:.6f}".format(train_err / train_batches)+"\n"
        email_content=email_content+"  training accuracy:\t\t{:.2f} %".format(train_acc / train_batches * 100)+"\n"       
        email_content=email_content+"  test loss:\t\t\t{:.6f}".format(test_err / test_batches)+"\n"
        email_content=email_content+"  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100)+"\n"
        email_content=email_content+"  test f1 score:\t\t{:.2f} %".format(test_f1_score*100)+"\n"
        email_content=email_content+"  max  f1 score:\t\t{:.2f} %".format(f1_score_max*100)+"\n"
        email_content=email_content+"  f1 max num epochs:\t\t    "+str(f1_max_num_epochs)+"\n"
        
        #each 50 epoches,save picture 
        if(epoch%50==0 and epoch!=0):
            num_epochs=range(epoch+1)
            num_epochs=[i+1 for i in num_epochs]
            #save Accuracy picture(train and test)
            model.save_plt(x=num_epochs,y=test_acc_listplt,label='test accuracy',title="Train And Test Accuracy",ylabel_name="accuracy(%)",save_path=model.save_picAcc_path,twice=True,x2=num_epochs,y2=train_acc_listplt,label2='train accuracy',showflag=False)
            #save F1-SCORE picture 
            model.save_plt(num_epochs,f1_score_listplt,'f1-score',"F1-SCORE","f1-score(%)",model.save_picF1_path,showflag=False)
            #save loss picture(train)
            model.save_plt(num_epochs,train_loss_listplt,'train loss',"Train Loss","loss",model.save_lossTrain_path,showflag=False)
            #save loss picture(test)
            model.save_plt(num_epochs,test_loss_listplt,'test loss',"Test Loss","loss",model.save_lossTest_path,showflag=False)
           
    """
    7.test model
    """
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in sdp_pro.iterate_minibatches(testing_word_pos_vec3D, \
      testing_label_1hot, testing_sen_length, test_left_sdp_length, model.batch_size, shuffle=False):
        inputs, targets, mask_sen_length, left_sdp_length = batch
        mark_input=model.mask(mask_sen_length,model.batch_size)
        err, acc, test_predicted_classid = val_fn(inputs, targets, mark_input, left_sdp_length, np.reshape(mask_sen_length,(-1,1)), model.negative_loss_alpha, model.negative_loss_lamda)
        test_err += err
        test_acc += acc
        test_batches += 1
    
    #F1 value
    mark_input=model.mask(testing_sen_length,len(testing_sen_length))
    err, acc, test_predicted_classid=val_fn(testing_word_pos_vec3D, testing_label_1hot, mark_input, test_left_sdp_length, np.reshape(testing_sen_length,(-1,1)), model.negative_loss_alpha, model.negative_loss_lamda)
    #confusion matrix
    test_con_mat=confusion_matrix(testing_label,test_predicted_classid)
    #support
    _,_,_,support=precision_recall_fscore_support(testing_label,test_predicted_classid)   
    #computer F1_Score
    test_f1_score=model.accuracy_f1(test_con_mat,support)
    
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    print("  test f1 score:\t\t{:.2f} %".format(test_f1_score*100))
    print("  max  f1 score:\t\t{:.2f} %".format(f1_score_max*100))
    print("  f1 max num epochs:\t\t"+str(f1_max_num_epochs))
    
    #save the "testing_label" and "test_predicted_classid"
    ture_predicted_save=np.concatenate((np.reshape(test_predicted_classid,(len(test_predicted_classid),1)),np.reshape(testing_label,(len(testing_label),1))),axis=1)
    ture_pre_save_txt=open("../testResult/ture_predicted_save.txt","w")
    for i in range(len(ture_predicted_save)):
        ture_pre_save_txt.write(str(ture_predicted_save[i][0])+" "+str(ture_predicted_save[i][1]))
        ture_pre_save_txt.write("\n")
    ture_pre_save_txt.close()
    
    """
    8.picture display
    """
    num_epochs=range(model.num_epochs)
    num_epochs=[i+1 for i in num_epochs]
    #save Accuracy picture(train and test)
    model.save_plt(x=num_epochs,y=test_acc_listplt,label='test accuracy',title="Train And Test Accuracy",ylabel_name="accuracy(%)",save_path=model.save_picAcc_path,twice=True,x2=num_epochs,y2=train_acc_listplt,label2='train accuracy')
    #save F1-SCORE picture 
    model.save_plt(num_epochs,f1_score_listplt,'f1-score',"F1-SCORE","f1-score(%)",model.save_picF1_path)
    #save loss picture(train)
    model.save_plt(num_epochs,train_loss_listplt,'train loss',"Train Loss","loss",model.save_lossTrain_path)
    #save loss picture(test)
    model.save_plt(num_epochs,test_loss_listplt,'test loss',"Test Loss","loss",model.save_lossTest_path)
    
    """
    9.send result to my email
    """
#    #Email content
#    email_content=email_content+"Final results:"+"\n"
#    email_content=email_content+"  test loss:\t\t\t{:.6f}".format(test_err / test_batches)+"\n"
#    email_content=email_content+"  test accuracy:\t\t{:.2f} %".format(
#            test_acc / test_batches * 100)+"\n"
#    email_content=email_content+"  test f1 score:\t\t{:.2f} %".format(test_f1_score*100)+"\n"
#    email_content=email_content+"  max f1 score:\t\t{:.2f} %".format(f1_score_max*100)+"\n"
#    email_content=email_content+"  f1 max num epochs:"+str(f1_max_num_epochs)+"\n"
#    
#
#    msg = MIMEMultipart()
#    msg['Subject'] = subject
#    msg['From'] = msg_from
#    msg['To'] = msg_to
#    
#    #Text Content
#    msg.attach(MIMEText(email_content, 'plain', 'utf-8'))
#    
#    #Construct attachment 1 and send the picture file in the current directory
#    att1 = MIMEText(open(model.save_picAcc_path, 'rb').read(), 'base64', 'utf-8')
#    att1["Content-Type"] = 'application/octet-stream'
#    # filename in email display
#    att1["Content-Disposition"] = 'attachment; filename="train and test accuracy.jpg"'
#    msg.attach(att1)
#    
#    #Construct attachment 2 and send the picture file in the current directory
#    att1 = MIMEText(open(model.save_picF1_path, 'rb').read(), 'base64', 'utf-8')
#    att1["Content-Type"] = 'application/octet-stream'
#    # filename in email display
#    att1["Content-Disposition"] = 'attachment; filename="f1-score.jpg"'
#    msg.attach(att1)
#    try:
#        s = smtplib.SMTP_SSL("smtp.qq.com",465)
#        s.login(msg_from, passwd)
#        s.sendmail(msg_from, msg_to, msg.as_string())
#        print "发送成功"
#    except s.SMTPException,e:
#        print "发送失败"
#    finally:
#        s.quit()
    
    """
    10.save result
    """
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
    result_file.write("f1_score_max="+str(f1_score_max)+",f1_max_num_epochs="+str(f1_max_num_epochs)+"\n")
    result_file.write("\n")
    result_file.write(email_content) 
    result_file.close()
    