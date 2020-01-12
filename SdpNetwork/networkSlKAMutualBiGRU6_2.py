#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 18:23:40 2018

mutual learning+KAS+stimulation loss(exp()*log(1+exp(-y^+)+exp(y^-)))

SSL:stimulative=1+coding_dist_true2inf-true_pre
    loss=stimulative*T.log(1+y_pre2true+y_pre2false)

Load the "./model/model_mutual.npz"

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
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix

from pretreatMent.process_data import Process_data
from CustomLayers import MarginLossLayer
from CustomLoss import SSL_Mutual2,kl_loss_compute2
from piGRU.DotSofTraLayers import SelfAttEntRootLayer,SelfAttEntRootLayer2,SelfAttEntRootLayer3,SelfAttEntRootLayerVariant,SelfAttEntRootLayer_emp

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
        self.num_steps = 107
        #the number of epoch(one epoch=N iterations)
        self.num_epochs = 110
        #the number of class
        self.num_classes = 19
        #the number of GRU units?
        self.gru_size = 250  #use in gru
        #dropout probability
        self.keep_prob_input=0.4                #use in input
        self.keep_prob_gru_output=0.4           #use in gru
        self.keep_prob_cnn=0.5                  #use in cnn
        self.keep_prob_cnn_gur_output=0.5       #use in output
        # the number of entity pairs of each batch during training or testing
        self.batch_size = 100
        #learning rate
        self.learning_rate_sgd=0.005
        #input shape
        self.input_shape=(None,107,340)
        #mask shape
        self.mask_shape=(None,107)
        # All gradients above this will be clipped
        self.grad_clip = 5
        #l2_loss
        self.l2_loss=1e-4
        # Choose "pi" or "                                                                       tempens" or "ordinary"
        self.network_type="ordinary"
        
        """
        GRU parameter
        """     
        self.gate_parameters = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
            b=lasagne.init.Constant())
        
        self.cell_parameters = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
            # Setting W_cell to None denotes that no cell connection will be used.
            W_cell=lasagne.init.Normal(), b=lasagne.init.Constant(),
            # By convention, the cell nonlinearity is tanh in an LSTM.
            nonlinearity=lasagne.nonlinearities.tanh)
        
        """Pi_model and Tempens model""" 
        # Ramp learning rate and unsupervised loss weight up during first n epochs.
        self.rampup_length=1
        # Ramp learning rate and Adam beta1 down during last n epochs.
        self.rampdown_length=30
        # Unsupervised loss maximum (w_max in paper). Set to 0.0 -> supervised loss only.
        self.scaled_unsup_weight_max=1.0
        # Maximum learning rate.
        self.learning_rate_max=0.005
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
        self.save_picAllAcc_path="../result/test all accuracy.jpg"
        self.save_picAllRec_path="../result/test all recall.jpg"
        #save train loss picture path
        self.save_lossTrain_path="../result/train-loss.jpg"
        #save test loss picture path
        self.save_lossTest_path="../result/test-loss.jpg"
        #save result file
        self.save_result="../result/result.txt"
        
        """Negtive loss"""
        self.negative_loss_alpha=np.float32(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.82375]))
        self.negative_loss_lamda=1
        
        """Self Attention"""
        self.input_shape_att=(None,340)
        self.attention_size2=10
        
    def bulit_gru(self,input_var=None,mask_var=None,input_root=None,input_e1=None,input_e2=None):
        """
        Bulit the GRU network
        """
        #inputlayer
        l_in=lasagne.layers.InputLayer(shape=self.input_shape,input_var=input_var,name="l_in")
       
        #mask layer
        l_mask=lasagne.layers.InputLayer(shape=self.mask_shape,input_var=mask_var,name="l_mask")
   
        #inpute dropout
        l_input_drop=lasagne.layers.DropoutLayer(l_in,p=self.keep_prob_input)
        
        """Input attention entity and root"""
        l_in_root=lasagne.layers.InputLayer(shape=self.input_shape_att,input_var=input_root,name="l_in_root")
        l_in_e1=lasagne.layers.InputLayer(shape=self.input_shape_att,input_var=input_e1,name="l_in_e1")
        l_in_e2=lasagne.layers.InputLayer(shape=self.input_shape_att,input_var=input_e2,name="l_in_e2")

       #Two GRU forward
        l_gru_forward=lasagne.layers.GRULayer(\
            l_input_drop,num_units=self.gru_size,grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters,updategate=self.gate_parameters,
            hidden_update=self.cell_parameters,learn_init=True,
            only_return_final=False,name="l_gru_forward1")

        l_gru_forward=lasagne.layers.GRULayer(\
            l_gru_forward,num_units=self.gru_size,grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters,updategate=self.gate_parameters,
            hidden_update=self.cell_parameters,learn_init=True,
            only_return_final=False,name="l_gru_forward2")

        
        #Two GRU backward
        l_gru_backward=lasagne.layers.GRULayer(\
            l_input_drop,num_units=self.gru_size,grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters,updategate=self.gate_parameters,
            hidden_update=self.cell_parameters,learn_init=True,
            only_return_final=False,backwards=True,name="l_gru_backward1")
 
        l_gru_backward=lasagne.layers.GRULayer(\
            l_gru_backward,num_units=self.gru_size,grad_clipping=self.grad_clip,
            resetgate=self.gate_parameters,updategate=self.gate_parameters,
            hidden_update=self.cell_parameters,learn_init=True,
            only_return_final=False,backwards=True,name="l_gru_backward2")
        
        
        #Merge forward layers and backward layers
        l_merge=lasagne.layers.ElemwiseSumLayer([l_gru_forward,l_gru_backward])

        #Keyword-Attention
        l_alphas=SelfAttEntRootLayer((l_in,l_in_e1,l_in_e2,l_in_root,l_merge),attention_size2=self.attention_size2)
        l_self_att=SelfAttEntRootLayer_emp((l_alphas,l_merge))
##        #when you need order in [Fe1,Fe2,Froot],you can use this line.
##        l_kas=SelfAttEntRootLayer3((l_self_att,l_merge))
        
        #output dropout
        l_merge_drop=lasagne.layers.DropoutLayer(l_self_att,p=self.keep_prob_gru_output)
        
        l_merge_fc = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
            l_merge_drop,
            num_units=250,
            W=lasagne.init.GlorotUniform(gain=1.0), b=lasagne.init.Constant(1.),
            nonlinearity=lasagne.nonlinearities.selu))
               
        #Margin loss
        #init w_classes: it's format is (class_number,network_output[1])
        w_classes_init=np.sqrt(6.0/(19+200))
        l_out_margin=MarginLossLayer(l_merge_fc,w_classes=lasagne.init.Uniform(w_classes_init), class_number=19)
     
        return l_out_margin,l_in,l_mask,l_out_margin,l_alphas

    def mask(self,mask_var,batch_size):
        """
        mask input:it's size is (batch_size,None)
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
                return 0,0,0
                
        accuracy=accuracy/9.0
        recall=recall/9.0
        f1=(2*accuracy*recall)/(accuracy+recall)
        return f1,accuracy,recall
    
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
    #set seed
    seed=np.random.RandomState(8)
    lasagne.random.set_rng(seed)
    
    """
    1.Loading data:training data and test data
    """
    print("Loading data")
    #1.Class:Process_data() and init the dict_word_vec
    pro_data=Process_data()
    pro_data.dict_word2vec()
    pro_data.label2id_init()
    
    #2.traing_word_pos_vec3D:training data
    pro_data.training_word_pos_vec3D,pro_data.training_sen_length,train_sen_list2D=\
      pro_data.embedding_lookup(pro_data.training_word_vec3D,\
      pro_data.training_word_pos_vec3D,pro_data.train_sen_store_filename,\
      pro_data.training_e1_e2_pos_filename,pro_data.training_sen_number)    
    training_word_pos_vec3D=np.float32(pro_data.training_word_pos_vec3D)
    training_sen_length=np.int32(np.array(pro_data.training_sen_length))
    
    #3.testing_word_pos_vec3D:testing data
    pro_data.testing_word_pos_vec3D,pro_data.testing_sen_length,test_sen_list2D=pro_data.embedding_lookup(pro_data.testing_word_vec3D,\
      pro_data.testing_word_pos_vec3D,pro_data.test_sen_store_filename,\
      pro_data.testing_e1_e2_pos_filename,pro_data.testing_sen_number)
    testing_word_pos_vec3D=np.float32(pro_data.testing_word_pos_vec3D)
    testing_sen_length=np.int32(np.array(pro_data.testing_sen_length))
        
    #4.training label:8000
    pro_data.training_label=pro_data.label2id_in_data(pro_data.train_label_store_filename,\
      pro_data.training_label)
    training_label=np.int32(pro_data.training_label)
        
    #5.testing label:2717
    pro_data.testing_label=pro_data.label2id_in_data(pro_data.test_label_store_filename,\
      pro_data.testing_label)
    testing_label=np.int32(pro_data.testing_label)
    
    #6.label id value: Change the label to id.And 10 classes number(0-9)
    label2id=pro_data.label2id
    
    #7.One-hot encode
    training_label_1hot=pro_data.label2id_1hot(training_label,label2id)
    training_label_1hot=np.int32(training_label_1hot)
    
    testing_label_1hot=pro_data.label2id_1hot(testing_label,label2id)    
    testing_label_1hot=np.int32(testing_label_1hot)

    #9.embedding root,e1 and e2
    train_root_embedding,train_e1_embedding,train_e2_embedding=\
        pro_data.embedding_looking_root_e1_e2(pro_data.e1_sdp_train_file,pro_data.e2_sdp_train_file,pro_data.training_sen_number,train_sen_list2D)
    
    test_root_embedding,test_e1_embedding,test_e2_embedding=\
        pro_data.embedding_looking_root_e1_e2(pro_data.e1_sdp_test_file,pro_data.e2_sdp_test_file,pro_data.testing_sen_number,test_sen_list2D)
  
    #8.mask_train_input:unsupervised.
    #Get in pro_data.mask_train_input()
    # Random shuffle.
    indices = np.arange(len(training_word_pos_vec3D))
    np.random.shuffle(indices)
    training_word_pos_vec3D = training_word_pos_vec3D[indices]
    training_sen_length = training_sen_length[indices]
    training_label_1hot=training_label_1hot[indices]
    train_root_embedding=train_root_embedding[indices]
    train_e1_embedding=train_e1_embedding[indices]
    train_e2_embedding=train_e2_embedding[indices]
    
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
      
    #negative loss
    negative_loss_alpha=T.fvector("negative_loss_alpha")
    negative_loss_lamda=T.fscalar("negative_loss_lamda") 
    
    #Keywords-attention
    input_root=T.fmatrix("input_root")
    input_e1=T.fmatrix("input_e1")
    input_e2=T.fmatrix("input_e2")
    
    unsup_weight_var = T.scalar('unsup_weight')
    
    
    """
    2.
    Bulit GRU network
    ADAM
    """
    gru_network,l_in,l_mask,l_gru_forward,l_alphas=model.bulit_gru(input_var,mask_var,input_root,input_e1,input_e2)
    gru_network2,_,_,_,_=model.bulit_gru(input_var,mask_var,input_root,input_e1,input_e2)
    
#    #And load them again later on like this:
#    with np.load('../model/model_mutual.npz') as f:
#         param_values = [f['arr_%d' % i] for i in range(len(f.files))]
#    lasagne.layers.set_all_param_values(gru_network, param_values)
#    with np.load('../model/model_mutual2.npz') as f:
#         param_values = [f['arr_%d' % i] for i in range(len(f.files))]
#    lasagne.layers.set_all_param_values(gru_network2, param_values)

    
    #mask_train_input: where "1" is pass. where "0" isn't pass.
    mask_train_input=pro_data.mask_train_input(training_label,num_labels=model.num_labels)
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(gru_network)
    prediction2 = lasagne.layers.get_output(gru_network2)
    l_gru = lasagne.layers.get_output(l_gru_forward)
    l_split = lasagne.layers.get_output(l_alphas)
    
#    loss = Negative_loss(prediction, target_var, negative_loss_alpha, negative_loss_lamda)
    loss,beta_false,beta_score, st_loss = SSL_Mutual2(prediction, target_var)
    loss2,_,_, st_loss2 = SSL_Mutual2(prediction2, target_var)
#    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)

    #Pi model loss
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
        loss2 = T.mean(loss2, dtype=theano.config.floatX)

        
    #regularization:L1,L2
    l2_penalty = lasagne.regularization.regularize_network_params(gru_network, lasagne.regularization.l2) * model.l2_loss
    loss=loss+l2_penalty
    l2_penalty2 = lasagne.regularization.regularize_network_params(gru_network2, lasagne.regularization.l2) * model.l2_loss
    loss2=loss2+l2_penalty2
    
    #KL-LOSS
    l_mul=unsup_weight_var*kl_loss_compute2(prediction,prediction2,target_var)
    loss=loss+l_mul
    l_mul2=unsup_weight_var*kl_loss_compute2(prediction2,prediction,target_var)
    loss2=loss2+l_mul2
    
    train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), T.argmax(target_var, axis=1)),
                      dtype=theano.config.floatX)
    train_acc2 = T.mean(T.eq(T.argmax(prediction2, axis=1), T.argmax(target_var, axis=1)),
                      dtype=theano.config.floatX)

    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(gru_network, trainable=True)
    updates = lasagne.updates.adam(
            loss, params, learning_rate=learning_rate_var, beta1=adam_beta1_var)
    params2 = lasagne.layers.get_all_params(gru_network2, trainable=True)
    updates2 = lasagne.updates.adam(
            loss2, params2, learning_rate=learning_rate_var, beta1=adam_beta1_var)
   
    
#    updates_sgd = lasagne.updates.sgd(
#            loss, params, learning_rate=learning_rate_var)

    """
    3.test loss and accuracy
    """
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(gru_network, deterministic=True)
    test_prediction2 = lasagne.layers.get_output(gru_network2, deterministic=True)

#    test_loss = Negative_loss(test_prediction,target_var,negative_loss_alpha,negative_loss_lamda)
#    test_loss=lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss,_,_,_ = SSL_Mutual2(test_prediction,target_var)
    test_loss = T.mean(test_loss,dtype=theano.config.floatX)
    test_loss2,_,_,_ = SSL_Mutual2(test_prediction2,target_var)
    test_loss2 = T.mean(test_loss2,dtype=theano.config.floatX)
 
    
    # As a bonus, also create an expression for the classification accuracy:
    test_predicted_classid=T.argmax(test_prediction, axis=1)
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)),
                      dtype=theano.config.floatX)
    test_predicted_classid2=T.argmax(test_prediction2, axis=1)
    test_acc2 = T.mean(T.eq(T.argmax(test_prediction2, axis=1), T.argmax(target_var, axis=1)),
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
        train_fn = theano.function([input_var, target_var, mask_var, learning_rate_var, adam_beta1_var, negative_loss_alpha, negative_loss_lamda, input_root, input_e1, input_e2, unsup_weight_var], [loss,train_acc,beta_false,beta_score,l_mul,st_loss], updates=updates, on_unused_input='warn')
        train_fn2 = theano.function([input_var, target_var, mask_var, learning_rate_var, adam_beta1_var, negative_loss_alpha, negative_loss_lamda, input_root, input_e1, input_e2, unsup_weight_var], [loss2,train_acc2,l_mul2,st_loss2], updates=updates2, on_unused_input='warn')

    # Compile a second function computing the validation loss and accuracy and F1-score:
    val_fn = theano.function([input_var, target_var, mask_var, negative_loss_alpha, negative_loss_lamda, input_root, input_e1, input_e2], [test_loss, test_acc, test_predicted_classid, l_gru, l_split], on_unused_input='warn')
    val_fn2 = theano.function([input_var, target_var, mask_var, negative_loss_alpha, negative_loss_lamda, input_root, input_e1, input_e2], [test_loss2, test_acc2, test_predicted_classid2], on_unused_input='warn')
    
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
    #accuracy
    accuracy_listplt=[]
    #recall
    recall_listplt=[]
    #Max F1-SCORE
    f1_score_max=0
    f1_score_max2=0
    #Max num_epoch in F1-SCORE
    f1_max_num_epochs=0
    
    #Train loss
    train_loss_listplt=[]
    #Test loss
    test_loss_listplt=[]
        
    wrong_pre_listepoch=[]
    true_pre_listepoch=[]
    l_mul_listplt=[]
    l_mul2_listplt=[]
    stimulative_listplt=[]
    stimulative2_listplt=[]
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
        train_err2 = 0
        train_acc2 =0
        train_pi_loss=0
        train_batches = 0
        start_time = time.time()
        
        l_mul=0
        l_mul2=0
        stimulative=0
        stimulative2=0
        #Pi model:run batch
        if model.network_type=="pi":
            for batch in pro_data.iterate_minibatches_pi(training_word_pos_vec3D, \
              training_label_1hot, training_sen_length, model.batch_size, mask_train_input, shuffle=True):
                inputs, targets, mask_sen_length, mask_train = batch
                mark_input=model.mask(mask_sen_length,model.batch_size)
                err ,acc, pi_loss = train_fn(inputs, targets, mark_input, inputs, mask_train, unsup_weight, learning_rate, adam_beta1)
                train_err+=err
                train_acc+=acc
                train_batches += 1
        elif model.network_type=="tempens":
        #Tempens model:run batch
            for batch in pro_data.iterate_minibatches_tempens(training_word_pos_vec3D, \
              training_label_1hot, training_sen_length, model.batch_size, mask_train_input, training_targets, shuffle=True):
                aa_inputs, targets, mask_sen_length, mask_train, z_targets, indices = batch
                aa_mark_input=model.mask(mask_sen_length,model.batch_size)
                err ,acc, prediction = train_fn(aa_inputs, targets, aa_mark_input, z_targets, mask_train, unsup_weight, learning_rate, adam_beta1)
                for i, j in enumerate(indices):
                    epoch_predictions[j] = prediction[i] # Gather epoch predictions.
                    epoch_execmask[j] = 1.0
                train_err+=err
                train_acc+=acc
                train_batches += 1
        else:
            #Ordinary model
            wrong_pre_list=[]
            true_pre_list=[]
            for batch in pro_data.iterate_minibatches_inputAttRootE1E2(training_word_pos_vec3D, \
              training_label_1hot, training_sen_length, model.batch_size, train_root_embedding, train_e1_embedding, train_e2_embedding, shuffle=True):
                aa_inputs, targets, mask_sen_length, input_root, input_e1, input_e2 = batch
                aa_mark_input=model.mask(mask_sen_length,model.batch_size)
                err ,acc, wrong_pre, true_pre, mul, st_loss = train_fn(aa_inputs, targets, aa_mark_input, learning_rate, adam_beta1, model.negative_loss_alpha, model.negative_loss_lamda, input_root, input_e1, input_e2, unsup_weight)                
                err2 ,acc2, mul2, st_loss2 = train_fn2(aa_inputs, targets, aa_mark_input, learning_rate, adam_beta1, model.negative_loss_alpha, model.negative_loss_lamda, input_root, input_e1, input_e2, unsup_weight)                
                
                train_err+=err
                train_acc+=acc
                train_err2+=err2
                train_acc2+=acc2
                train_batches += 1
        
                wrong_pre_list.append(wrong_pre)
                true_pre_list.append(true_pre)
                
                l_mul+=mul
                l_mul2+=mul2
                stimulative+=np.mean(st_loss)
                stimulative2+=np.mean(st_loss2)
                
            wrong_pre_listepoch.append(wrong_pre_list)
            true_pre_listepoch.append(true_pre_list)
            
        #train accuracy
        train_acc_listplt.append(train_acc / train_batches * 100)
        #train loss
        train_loss_listplt.append(train_err / train_batches)
        
        l_mul_listplt.append(l_mul/train_batches)
        l_mul2_listplt.append(l_mul2/train_batches)
        stimulative_listplt.append(stimulative/train_batches)
        stimulative2_listplt.append(stimulative2/train_batches)
        
        # Each epoch training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_err2 = 0
        test_acc2 = 0
        test_batches = 0
        l_att_matrix_epoch=[]
        
        test_predicted_classi_all=np.array([0])
        test_predicted_classi_all2=np.array([0])
        for batch in pro_data.iterate_minibatches_inputAttRootE1E2(testing_word_pos_vec3D, \
          testing_label_1hot, testing_sen_length, model.batch_size, test_root_embedding, test_e1_embedding, test_e2_embedding, shuffle=False):
            inputs, targets, mask_sen_length, input_root, input_e1, input_e2 = batch
            mark_input=model.mask(mask_sen_length,model.batch_size)
            err, acc, test_predicted_classid, _, l_split = val_fn(inputs, targets, mark_input, model.negative_loss_alpha, model.negative_loss_lamda, input_root, input_e1, input_e2)
            err2, acc2, test_predicted_classid2 = val_fn2(inputs, targets, mark_input, model.negative_loss_alpha, model.negative_loss_lamda, input_root, input_e1, input_e2)

            test_err += err
            test_acc += acc
            test_err2 += err2
            test_acc2 += acc2
            test_batches += 1
            
            test_predicted_classi_all=np.concatenate((test_predicted_classi_all,test_predicted_classid))
            test_predicted_classi_all2=np.concatenate((test_predicted_classi_all2,test_predicted_classid2))
            
            l_att_matrix_epoch.append(l_split)
            
        test_predicted_classid=np.array(test_predicted_classi_all[1:-83])
        test_predicted_classid2=np.array(test_predicted_classi_all2[1:-83])
        
        #test accuracy
        test_acc_listplt.append(test_acc / test_batches * 100)
        #test loss
        test_loss_listplt.append(test_err / test_batches)
        
        #update "ensemble_prediction(Z)" and "training_targets"
        if model.network_type == 'tempens':
            # Basic mode.
            ensemble_prediction = (model.prediction_decay * ensemble_prediction) + (1.0 - model.prediction_decay) * epoch_predictions
            training_targets = ensemble_prediction / (1.0 - model.prediction_decay ** ((epoch - 0) + 1.0))

#        #F1 value
#        mark_input=model.mask(testing_sen_length,len(testing_sen_length))
#        err, acc, test_predicted_classid,_=val_fn(testing_word_pos_vec3D, testing_label_1hot, mark_input, model.negative_loss_alpha, model.negative_loss_lamda, test_root_embedding, test_e1_embedding, test_e2_embedding)
        #confusion matrix
        test_con_mat=confusion_matrix(testing_label[:-83],test_predicted_classid)
        #support
        _,_,_,support=precision_recall_fscore_support(testing_label[:-83],test_predicted_classid)   
        #computer F1_Score
        test_f1_score,test_accuracy,test_recall=model.accuracy_f1(test_con_mat,support)

        test_con_mat=confusion_matrix(testing_label[:-83],test_predicted_classid2)
        #support
        _,_,_,support=precision_recall_fscore_support(testing_label[:-83],test_predicted_classid2)   
        #computer F1_Score
        test_f1_score2,test_accuracy2,test_recall2=model.accuracy_f1(test_con_mat,support)

        
        f1_score_listplt.append(test_f1_score*100)
        accuracy_listplt.append(test_accuracy*100)
        recall_listplt.append(test_recall*100)
        
        
        #max f1-score
        if(f1_score_max<test_f1_score):
            f1_score_max=test_f1_score
            f1_max_num_epochs=epoch+1
        
        #max f1-score
        if(f1_score_max2<test_f1_score2):
            f1_score_max2=test_f1_score2
            
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, model.num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.2f} %".format(train_acc / train_batches * 100))
        print("  training loss2:\t\t{:.6f}".format(train_err2 / train_batches))
        print("  training accuracy2:\t\t{:.2f} %".format(train_acc2 / train_batches * 100))

        #Testing loss and accuracy
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))
        print("  test f1 score:\t\t{:.2f} %".format(test_f1_score*100))
        print("  test all accuracy:\t\t{:.2f} %".format(test_accuracy*100))
        print("  test all recall:\t\t{:.2f} %".format(test_recall*100))
        
        print("  max  f1 score:\t\t{:.2f} %".format(f1_score_max*100))
        print("  max  f1 score2:\t\t{:.2f} %".format(f1_score_max2*100))

        print("  learning rate:\t\t{:.6f}".format(learning_rate))        
        print("  f1 max num epochs:\t\t    "+str(f1_max_num_epochs))
        
        #Email content
        email_content=email_content+"Epoch {} of {} took {:.3f}s".format(
            epoch + 1, model.num_epochs, time.time() - start_time)+"\n"
        email_content=email_content+"  training loss:\t\t{:.6f}".format(train_err / train_batches)+"\n"
        email_content=email_content+"  training accuracy:\t\t{:.2f} %".format(train_acc / train_batches * 100)+"\n"       
        email_content=email_content+"  training loss2:\t\t{:.6f}".format(train_err2 / train_batches)+"\n"
        email_content=email_content+"  training accuracy2:\t\t{:.2f} %".format(train_acc2 / train_batches * 100)+"\n"       
        email_content=email_content+"  test loss:\t\t\t{:.6f}".format(test_err / test_batches)+"\n"
        email_content=email_content+"  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100)+"\n"
        email_content=email_content+"  test f1 score:\t\t{:.2f} %".format(test_f1_score*100)+"\n"
        email_content=email_content+"  test all accuracy:\t\t{:.2f} %".format(test_accuracy*100)+"\n"
        email_content=email_content+"  test all recall:\t\t{:.2f} %".format(test_recall*100)+"\n"
        email_content=email_content+"  max  f1 score:\t\t{:.2f} %".format(f1_score_max*100)+"\n"
        email_content=email_content+"  max  f1 score2:\t\t{:.2f} %".format(f1_score_max2*100)+"\n"
        email_content=email_content+"  f1 max num epochs:\t\t    "+str(f1_max_num_epochs)+"\n"
        
#        if(epoch%10==0 and epoch!=0):
#            if(f1_score_listplt[-1]<f1_score_listplt[-10]):
#                learning_rate=learning_rate/2.0
                
        #each 50 epoches,save picture 
        if(epoch%10==0 and epoch!=0):
            num_epochs=range(epoch+1)
            num_epochs=[i+1 for i in num_epochs]
            #save Accuracy picture(train and test)
            model.save_plt(x=num_epochs,y=test_acc_listplt,label='test accuracy',title="Train And Test Accuracy",ylabel_name="accuracy(%)",save_path=model.save_picAcc_path,twice=True,x2=num_epochs,y2=train_acc_listplt,label2='train accuracy',showflag=False)
            #save F1-SCORE picture 
            model.save_plt(num_epochs,f1_score_listplt,'f1-score',"F1-SCORE","f1-score(%)",model.save_picF1_path,showflag=False)
            #save Accuracy picture 
            model.save_plt(num_epochs,accuracy_listplt,'test all accuracy',"Test All Accuracy","accuracy(%)",model.save_picAllAcc_path,showflag=False)
            #save Recall picture 
            model.save_plt(num_epochs,recall_listplt,'test all recall',"Test All Recall","recall(%)",model.save_picAllRec_path,showflag=False)
            #save loss picture(train)
            model.save_plt(num_epochs,train_loss_listplt,'train loss',"Train Loss","loss",model.save_lossTrain_path,showflag=False)
            #save loss picture(test)
            model.save_plt(num_epochs,test_loss_listplt,'test loss',"Test Loss","loss",model.save_lossTest_path,showflag=False)
            #save loss KL()
            model.save_plt(num_epochs,l_mul_listplt,'kl',"KL","loss",save_path="../result/kl.jpg",showflag=False)
            model.save_plt(num_epochs,l_mul2_listplt,'kl2',"KL2","loss",save_path="../result/kl2.jpg",showflag=False)
            model.save_plt(num_epochs,stimulative_listplt,'stimulation',"Stimulation","loss",save_path="../result/Stimulation.jpg",showflag=False)
            model.save_plt(num_epochs,stimulative2_listplt,'stimulation2',"Stimulation2","loss",save_path="../result/Stimulation2.jpg",showflag=False)

#    """
#    7.test model
#    """
#    # After training, we compute and print the test error:
#    test_err = 0
#    test_acc = 0
#    test_batches = 0
#    l_att_matrix_epoch_list=[]
#    for batch in pro_data.iterate_minibatches_inputAttRootE1E2(testing_word_pos_vec3D, \
#      testing_label_1hot, testing_sen_length, model.batch_size, test_root_embedding, test_e1_embedding, test_e2_embedding, shuffle=False):
#        inputs, targets, mask_sen_lengthm, input_root, input_e1, input_e2 = batch
#        mark_input=model.mask(mask_sen_length,model.batch_size)
#        err, acc, test_predicted_classid, l_att_matrix = val_fn(inputs, targets, mark_input, model.negative_loss_alpha, model.negative_loss_lamda, input_root, input_e1, input_e2)
#        test_err += err
#        test_acc += acc
#        test_batches += 1
#        
#        l_att_matrix_epoch.append(l_att_matrix)
#        
#    l_att_matrix_epoch_list.append(l_att_matrix_epoch)
#    
#    #F1 value
#    mark_input=model.mask(testing_sen_length,len(testing_sen_length))
#    err, acc, test_predicted_classid,_=val_fn(testing_word_pos_vec3D, testing_label_1hot, mark_input, model.negative_loss_alpha, model.negative_loss_lamda, test_root_embedding, test_e1_embedding, test_e2_embedding)
#    #confusion matrix
#    test_con_mat=confusion_matrix(testing_label,test_predicted_classid)
#    #support
#    _,_,_,support=precision_recall_fscore_support(testing_label,test_predicted_classid)   
#    #computer F1_Score
#    test_f1_score,test_accuracy,test_recall=model.accuracy_f1(test_con_mat,support)
#    
#    print("Final results:")
#    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
#    print("  test accuracy:\t\t{:.2f} %".format(
#        test_acc / test_batches * 100))
#    print("  test f1 score:\t\t{:.2f} %".format(test_f1_score*100))
#    print("  max  f1 score:\t\t{:.2f} %".format(f1_score_max*100))
#    print("  f1 max num epochs:\t\t"+str(f1_max_num_epochs))
#    
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
    #save Accuracy picture(train and test)
    model.save_plt(x=num_epochs,y=test_acc_listplt,label='test accuracy',title="Train And Test Accuracy",ylabel_name="accuracy(%)",save_path=model.save_picAcc_path,twice=True,x2=num_epochs,y2=train_acc_listplt,label2='train accuracy')
    #save F1-SCORE picture 
    model.save_plt(num_epochs,f1_score_listplt,'f1-score',"F1-SCORE","f1-score(%)",model.save_picF1_path)
    #save Accuracy picture 
    model.save_plt(num_epochs,accuracy_listplt,'test all accuracy',"Test All Accuracy","accuracy(%)",model.save_picAllAcc_path,showflag=False)
    #save Recall picture 
    model.save_plt(num_epochs,recall_listplt,'test all recall',"Test All Recall","recall(%)",model.save_picAllRec_path,showflag=False)
    #save loss picture(train)
    model.save_plt(num_epochs,train_loss_listplt,'train loss',"Train Loss","loss",model.save_lossTrain_path)
    #save loss picture(test)
    model.save_plt(num_epochs,test_loss_listplt,'test loss',"Test Loss","loss",model.save_lossTest_path)
    
    """
    9.send result to my email
    """
    #Email content
    email_content=email_content+"Final results:"+"\n"
    email_content=email_content+"  test loss:\t\t\t{:.6f}".format(test_err / test_batches)+"\n"
    email_content=email_content+"  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100)+"\n"
    email_content=email_content+"  test f1 score:\t\t{:.2f} %".format(test_f1_score*100)+"\n"
    email_content=email_content+"  max f1 score:\t\t{:.2f} %".format(f1_score_max*100)+"\n"
    email_content=email_content+"  f1 max num epochs:"+str(f1_max_num_epochs)+"\n"
   
    
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = msg_from
    msg['To'] = msg_to
    
    #Text Content
    msg.attach(MIMEText(email_content, 'plain', 'utf-8'))
    
    #Construct attachment 1 and send the picture file in the current directory
    att1 = MIMEText(open(model.save_picAcc_path, 'rb').read(), 'base64', 'utf-8')
    att1["Content-Type"] = 'application/octet-stream'
    # filename in email display
    att1["Content-Disposition"] = 'attachment; filename="train and test accuracy.jpg"'
    msg.attach(att1)
    
    #Construct attachment 2 and send the picture file in the current directory
    att1 = MIMEText(open(model.save_picF1_path, 'rb').read(), 'base64', 'utf-8')
    att1["Content-Type"] = 'application/octet-stream'
    # filename in email display
    att1["Content-Disposition"] = 'attachment; filename="f1-score.jpg"'
    msg.attach(att1)
    try:
        s = smtplib.SMTP_SSL("smtp.qq.com",465)
        s.login(msg_from, passwd)
        s.sendmail(msg_from, msg_to, msg.as_string())
        print "发送成功"
    except s.SMTPException,e:
        print "发送失败"
    finally:
        s.quit()
    
    """
    10.save result
    """
    result_file=open(model.save_result,"w")
    result_file.write("num_steps="+str(model.num_steps)+"\n")
    result_file.write("num_epochs="+str(model.num_epochs)+"\n")
    result_file.write("num_classes="+str(model.num_classes)+"\n")
    result_file.write("gru_size="+str(model.gru_size)+"\n")
    result_file.write("keep_prob_input="+str(model.keep_prob_input)+"\n")
    result_file.write("keep_prob_gru_output="+str(model.keep_prob_gru_output)+"\n")
    result_file.write("keep_prob_cnn="+str(model.keep_prob_cnn)+"\n")
    result_file.write("keep_prob_cnn_gur_output="+str(model.keep_prob_cnn_gur_output)+"\n")
    result_file.write("batch_size="+str(model.batch_size)+"\n")
    result_file.write("learning_rate_sgd="+str(model.learning_rate_sgd)+"\n")
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

#    #Optionally, you could now dump the network weights to a file like this:
#    np.savez('../model/model_mutual.npz', *lasagne.layers.get_all_param_values(gru_network))
#    np.savez('../model/model_mutual2.npz', *lasagne.layers.get_all_param_values(gru_network2))


