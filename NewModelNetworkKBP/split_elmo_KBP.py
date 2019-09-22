#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:14:54 2019

@author: llq
@function:
    1.split the train data and test data.
    2.split 4 data
"""

from dataElmoKBP import ELMO_KBP
import time
import numpy as np
import os



def merge_embedding(split_n, glove_path, elmo_path, merge_path, glove_name, elmo_name, merge_name):
    """
    function:
        1.merge the three embedding
    """
    for i in range(split_n):
        glove_file = os.path.join(glove_path, glove_name + str(i) + '.npz')
        elmo_file = os.path.join(elmo_path, elmo_name + str(i) + '.npy')
        merge_file = os.path.join(merge_path, merge_name + str(i) + '.npz')
        
        #glove data
        glove_data = np.load(glove_file)
        training_word_pos_vec3D = glove_data['training_word_pos_vec3D']
        training_sen_length = glove_data['training_sen_length']
        train_pos_vec = glove_data['train_pos_vec']
        train_sen_list2D = glove_data['train_sen_list2D']
        training_label_1hot = glove_data['training_label_1hot']
        train_root_embedding = glove_data['train_root_embedding']
        train_e1_embedding = glove_data['train_e1_embedding']
        train_e2_embedding = glove_data['train_e2_embedding']
        
        
        #elmo embedding
        elmo_embedding = np.load(elmo_file)
        
        #merge three embedding
        word_vec3D = np.concatenate((training_word_pos_vec3D, elmo_embedding, train_pos_vec), axis=2)
        
        data = dict(
            word_vec3D = word_vec3D,
            sen_length = training_sen_length,
            sen_list2D = train_sen_list2D,
            label_1hot = training_label_1hot,
            root_embedding = train_root_embedding,
            e1_embedding = train_e1_embedding,
            e2_embedding = train_e2_embedding)
        np.savez(merge_file, **data)
    
def split_data(split_n, glove_path, glove_name, train_len, word_pos_vec3D, sen_length, \
               pos_vec, sen_list2D, label_1hot, root_embedding, e1_embedding, e2_embedding):
    """
    function:
        1.split the data
    """
    for i in range(split_n):
        glove_file = os.path.join(glove_path, glove_name + str(i) + '.npz')
        if(i < split_n - 1):
            start = i * (train_len / split_n)
            end = (i + 1) * (train_len / split_n)
        else:
            start = i * (train_len / split_n)
            end = (i + 1) * (train_len / split_n) + (train_len % split_n)
        
        data = dict(
            training_word_pos_vec3D = word_pos_vec3D[start:end],
            training_sen_length = sen_length[start:end],
            train_pos_vec = pos_vec[start:end],
            train_sen_list2D = sen_list2D[start:end],
            training_label_1hot = label_1hot[start:end],
            train_root_embedding = root_embedding[start:end],
            train_e1_embedding = e1_embedding[start:end],
            train_e2_embedding = e2_embedding[start:end])
        np.savez(glove_file, **data)
        
if __name__ == "__main__":
    """
    1.split the train data
    """
    glove_path = './data/glove_embedding'
    elmo_path = './data/elmo_embedding'
    merge_path = './data/merge_embedding'
    indices_file = './data/indices.npy'
    split_n = 4
    
    elmo_kbp = ELMO_KBP()
    start_time = time.time()
      
#    elmo_kbp.dict_word2vec()
    elmo_kbp.label2id_init()
    print("load the dict word2vec: %f s" % (time.time() - start_time))
    
    #traing_word_pos_vec3D:training data
    
#    training_word_pos_vec3D, train_pos_vec, training_sen_length,train_sen_list2D=\
#      elmo_kbp.embedding_lookup(elmo_kbp.train_sen_store_filename,\
#      elmo_kbp.training_e1_e2_pos_filename,elmo_kbp.training_sen_number)    
#    training_word_pos_vec3D=np.float32(training_word_pos_vec3D)
#    training_sen_length=np.int32(np.array(training_sen_length))
#    print("load the train glove embedding: %f s" % (time.time() - start_time)) 
#    
#    #4.training label
#    label2id = elmo_kbp.label2id
#    training_label = elmo_kbp.label2id_in_data(elmo_kbp.train_label_store_filename,\
#      elmo_kbp.training_sen_number)
#    training_label = np.int32(training_label)
#    training_label_1hot = elmo_kbp.label2id_1hot(training_label, label2id)
#    training_label_1hot = np.int32(training_label_1hot)
#    
#    #load the embedding of root, e1 and e2.
#    train_root_embedding, train_e1_embedding, train_e2_embedding = \
#        elmo_kbp.embedding_looking_root_e1_e2(elmo_kbp.e1_sdp_train_file,\
#        elmo_kbp.e2_sdp_train_file, elmo_kbp.training_sen_number, train_sen_list2D, elmo_kbp.train_elmo_file)
#    print("load the embedding of root, e1 and e2: %f s" % (time.time() - start_time)) 
#    
#    #shuffle
#    train_len = len(training_word_pos_vec3D)
#    indices = np.load(indices_file)
##    np.random.shuffle(indices)
#    training_word_pos_vec3D = training_word_pos_vec3D[indices]
#    training_sen_length = training_sen_length[indices]
#    train_pos_vec = train_pos_vec[indices]
#    training_label_1hot = training_label_1hot[indices]
#    train_root_embedding = train_root_embedding[indices]
#    train_e1_embedding = train_e1_embedding[indices]
#    train_e2_embedding = train_e2_embedding[indices]
#    
#    #split the data
#    for i in range(split_n):
#        glove_file = os.path.join(glove_path, 'train_glove_embedding_' + str(i) + '.npz')
#        if(i < 3):
#            start = i * (train_len / 4)
#            end = (i + 1) * (train_len / 4)
#        else:
#            start = i * (train_len / 4)
#            end = (i + 1) * (train_len / 4) + (train_len % 4)
#        
#        data = dict(
#            training_word_pos_vec3D = training_word_pos_vec3D[start:end],
#            training_sen_length = training_sen_length[start:end],
#            train_pos_vec = train_pos_vec[start:end],
#            train_sen_list2D = train_sen_list2D[start:end],
#            training_label_1hot = training_label_1hot[start:end],
#            train_root_embedding = train_root_embedding[start:end],
#            train_e1_embedding = train_e1_embedding[start:end],
#            train_e2_embedding = train_e2_embedding[start:end])
#        np.savez(glove_file, **data)
    
#    np.save(indices_file, indices)
 
    
#    #ELMO embedding
#    train_elmo_embedding = np.load(elmo_kbp.train_elmo_file)
#    indices = np.load(indices_file)
#    train_elmo_embedding = train_elmo_embedding[indices]
#    train_len = len(train_elmo_embedding)
#    
#    #split the data
#    for i in range(split_n):
#        elmo_file = os.path.join(elmo_path, 'train_elmo_embedding_' + str(i) + '.npy')
#        if(i < 3):
#            start = i * (train_len / 4)
#            end = (i + 1) * (train_len / 4)
#        else:
#            start = i * (train_len / 4)
#            end = (i + 1) * (train_len / 4) + (train_len % 4)
#        print start, end
#        np.save(elmo_file, train_elmo_embedding[start:end])

#    #train merge embedding
#    train_glove_name = 'train_glove_embedding_'
#    train_elmo_name = 'train_elmo_embedding_'
#    train_merge_name = 'train_merge_embedding_'
#    merge_embedding(split_n, glove_path, elmo_path, merge_path, train_glove_name, train_elmo_name, train_merge_name)
    
    """
    2.test data
    """
    #split the test glove embedding
    split_n = 2
    test_glove_name = 'test_glove_embedding_'
    
#    #testing_word_pos_vec3D:testing data
#    testing_word_pos_vec3D, test_pos_vec, testing_sen_length,test_sen_list2D=\
#      elmo_kbp.embedding_lookup(elmo_kbp.test_sen_store_filename,\
#      elmo_kbp.testing_e1_e2_pos_filename,elmo_kbp.testing_sen_number)
#    testing_word_pos_vec3D=np.float32(testing_word_pos_vec3D)
#    testing_sen_length=np.int32(np.array(testing_sen_length))
#    print("load the test glove embedding: %f s" % (time.time() - start_time)) 
#    
#    #5.testing label
#    label2id = elmo_kbp.label2id
#    testing_label=elmo_kbp.label2id_in_data(elmo_kbp.test_label_store_filename,\
#      elmo_kbp.testing_sen_number)
#    testing_label=np.int32(testing_label)
#    
#    testing_label_1hot = elmo_kbp.label2id_1hot(testing_label, label2id)    
#    testing_label_1hot = np.int32(testing_label_1hot)
#    
#    test_root_embedding, test_e1_embedding, test_e2_embedding=\
#        elmo_kbp.embedding_looking_root_e1_e2(elmo_kbp.e1_sdp_test_file,\
#        elmo_kbp.e2_sdp_test_file, elmo_kbp.testing_sen_number, test_sen_list2D, elmo_kbp.test_elmo_file)
#
#    #split the test data
#    test_len = len(testing_word_pos_vec3D)
#    split_data(split_n, glove_path, test_glove_name, test_len, testing_word_pos_vec3D, testing_sen_length, \
#               test_pos_vec, test_sen_list2D, testing_label_1hot, test_root_embedding, test_e1_embedding, test_e2_embedding)
    
    #ELMO embedding
    test_elmo_name = 'test_elmo_embedding_'
#    test_elmo_embedding = np.load(elmo_kbp.test_elmo_file)
#    test_len = len(test_elmo_embedding)
#    
#    #split the data
#    for i in range(split_n):
#        elmo_file = os.path.join(elmo_path, test_elmo_name + str(i) + '.npy')
#        if(i < split_n - 1):
#            start = i * (test_len / split_n)
#            end = (i + 1) * (test_len / split_n)
#        else:
#            start = i * (test_len / split_n)
#            end = (i + 1) * (test_len / split_n) + (test_len % split_n)
#        print start, end
#        np.save(elmo_file, test_elmo_embedding[start:end])
        
    #merge embedding
    test_merge_name = 'test_merge_embedding_'
    merge_embedding(split_n, glove_path, elmo_path, merge_path, test_glove_name, test_elmo_name, test_merge_name)
