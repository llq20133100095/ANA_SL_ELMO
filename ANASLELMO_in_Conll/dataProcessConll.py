#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2019.9.7 12：51

@author: llq
@function:
    1.process the Conll04 dataset
    2.concate the "glove embedding"
        and "Elmo embedding" and "Pos embedding"
"""
import os
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
import time

dir_path = os.path.dirname(__file__)


class ELMO_CONLL(object):

    def __init__(self):
        """ 1. download from internet """
        self.download_file = os.path.join(dir_path, "../data/conll04/conll04_relation/conll04_all.txt")

        """ 2. split other class """
        self.other_file = os.path.join(dir_path, "../data/conll04/conll04_relation/conll04_other.txt")
        self.no_other_file = os.path.join(dir_path, "../data/conll04/conll04_relation/conll04_no_other.txt")

        self.max_length_sen = 107

        """ 3.Word2vec """
        self.output_vector_filename = os.path.join(dir_path, "../data/conll04/conll04_relation_glove/conll04_glove_300.txt")
        self.dict_word_vec = {}
        self.vector_size = 300

        self.label2id_txt = os.path.join(dir_path, "../data/conll04/conll04_relation/label2id.txt")
        self.label2id = {}

        """ 4. training file """
        self.train_sen_store_filename = os.path.join(dir_path, "../data/conll04/conll04_relation_train_test/conll04_train_sen.txt")
        self.train_label_store_filename = os.path.join(dir_path, "../data/conll04/conll04_relation_train_test/conll04_train_label.txt")
        self.train_e1_e2_pos_filename = os.path.join(dir_path, "../data/conll04")

        """ 5. testing file """
        self.test_sen_store_filename = os.path.join(dir_path, "../data/conll04/conll04_relation_train_test/conll04_test_sen.txt")
        self.test_label_store_filename = os.path.join(dir_path, "../data/conll04/conll04_relation_train_test/conll04_test_label.txt")

    def split_other_class(self):
        """
        split other class
        """
        conll04_data = []
        other_file = open(self.other_file, "w")
        no_other_file = open(self.no_other_file, "w")

        with open(elmo_conll.download_file, "r") as file:
            content = file.readlines()
            n = len(content) - 1
            index = 0
            while index < n:
                conll04_data.append(content[index])
                if content[index] == "\n" and content[index + 1] == "\n":
                    for i in conll04_data:
                        other_file.write(i)
                    conll04_data = []
                    index += 1
                elif content[index] == "\n":
                    for i in conll04_data:
                        no_other_file.write(i)
                    conll04_data = []
                index += 1

        other_file.close()
        no_other_file.close()

    def process_no_other_data(self):
        conll04_data_list = []
        conll04_label_list = []

        # process the data, and get the one label
        is_label = False
        line_null = 0
        temp_list = []
        entity_pair_num = 0
        e1_pos = -1
        e2_pos = -1
        with open(self.no_other_file, "r") as file:
            while True:
                line = file.readline()
                if line:
                    if line == "\n":
                        is_label = not is_label
                        line_null += 1
                        entity_pair_num = 0
                        if len(temp_list) != 0 and line_null == 2:
                            if e1_pos < e2_pos:
                                temp_list.insert(e1_pos, "<e1>")
                                temp_list.insert(e1_pos + 2, "<\e1>")
                                temp_list.insert(e2_pos + 2, "<e2>")
                                temp_list.insert(e2_pos + 4, "<\e2>")
                            else:
                                temp_list.insert(e2_pos, "<e2>")
                                temp_list.insert(e2_pos + 2, "<\e2>")
                                temp_list.insert(e1_pos + 2, "<e1>")
                                temp_list.insert(e1_pos + 4, "<\e1>")
                            conll04_data_list.append(temp_list)
                            temp_list = []
                            e1_pos = -1
                            e2_pos = -1
                            line_null = 0
                    elif not is_label:
                        line = line.strip("\n").split("\t")
                        temp_list.append(line[5])
                    else:
                        if entity_pair_num > 0:
                            continue
                        line = line.strip("\n").split("\t")
                        conll04_label_list.append(line[2])
                        e1_pos = int(line[0])
                        e2_pos = int(line[1])
                        entity_pair_num += 1
                else:
                    break

        # save data
        no_other_save = open(os.path.join(dir_path, "../data/conll04/conll04_relation/conll04_no_other_sen.txt"), "w")
        no_othe_label_save = open(
            os.path.join(dir_path, "../data/conll04/conll04_relation/conll04_no_other_sen_label.txt"), "w")
        result_list = []
        for index, each_line in enumerate(conll04_data_list):
            temp_line = []
            temp_line2 = []
            for word in each_line:
                word = word.lower()
                word = word.split("/")
                temp_line += word
            for word in temp_line:
                word = word.split("-")
                temp_line2 += word
            no_other_save.write(" ".join(temp_line2))
            no_other_save.write("\n")
            result_list.append(temp_line2)
            no_othe_label_save.write(conll04_label_list[index] + "\n")
        no_other_save.close()
        no_othe_label_save.close()
        return result_list

    def process_other_data(self):
        conll04_data_list = []
        conll04_label_list = []
        temp_list = []
        e1_pos = -1
        e2_pos = -1
        entity_list = []

        with open(elmo_conll.other_file, "r") as f:
            while True:
                line = f.readline()
                if line:
                    if line == "\n":
                        if len(temp_list) != 0:
                            for index, pos in enumerate(entity_list):
                                if pos != "O":
                                    e1_pos = index
                                    break
                            for index in list(reversed(range(len(entity_list)))):
                                if entity_list[index] != "O":
                                    e2_pos = index
                                    break
                            if e1_pos == e2_pos:
                                temp_list = []
                                entity_list = []
                                e1_pos = -1
                                e2_pos = -1
                                continue
                            elif e1_pos < e2_pos:
                                temp_list.insert(e1_pos, "<e1>")
                                temp_list.insert(e1_pos + 2, "<\e1>")
                                temp_list.insert(e2_pos + 2, "<e2>")
                                temp_list.insert(e2_pos + 4, "<\e2>")
                            else:
                                temp_list.insert(e2_pos, "<e2>")
                                temp_list.insert(e2_pos + 2, "<\e2>")
                                temp_list.insert(e1_pos + 2, "<e1>")
                                temp_list.insert(e1_pos + 4, "<\e1>")
                            conll04_data_list.append(temp_list)
                            temp_list = []
                            entity_list = []
                            conll04_label_list.append("other")
                            e1_pos = -1
                            e2_pos = -1
                    else:
                        line = line.strip().split("\t")
                        temp_list.append(line[5])
                        entity_list.append(line[1])
                else:
                    break

        # save other data
        other_save = open(os.path.join(dir_path, "../data/conll04/conll04_relation/conll04_other_sen.txt"), "w")
        other_label_save = open(os.path.join(dir_path, "../data/conll04/conll04_relation/conll04_other_sen_label.txt"), "w")
        result_list = []
        for index, each_line in enumerate(conll04_data_list):
            temp_line = []
            temp_line2 = []
            for word in each_line:
                word = word.lower()
                word = word.split("/")
                temp_line += word
            for word in temp_line:
                word = word.split("-")
                temp_line2 += word
            other_save.write(" ".join(temp_line2))
            other_save.write("\n")
            result_list.append(temp_line2)
            other_label_save.write(conll04_label_list[index] + "\n")
        other_save.close()
        other_label_save.close()
        return result_list

    def get_data_in_file(self, data_file):
        ret_list = []
        with open(data_file, "r") as f:
            while True:
                line = f.readline().strip()
                if line:
                    ret_list.append(line.split(" "))
                else:
                    break
        return ret_list

    # elmo embedding
    def get_elmo_embedding(self, sen_list2D):
        #use the python3
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        for i in range(len(sen_list2D)):
            sub_value = self.max_length_sen - len(sen_list2D[i])
            sen_list2D[i] = sen_list2D[i] + ["blank"] * sub_value

        fin_embedding = np.zeros((len(sen_list2D), self.max_length_sen, 1024))
        start_time = time.time()

        elmo = ElmoEmbedder(options_file, weight_file)
        for i in range(len(sen_list2D)):
            print('iter: %d | use %f s time' % (i, time.time() - start_time))
            elmo_embedding, elmo_mask = elmo.batch_to_embeddings(sen_list2D[i:i+1])
            # select the last layer as embedding
            elmo_embedding = np.array(elmo_embedding[0][2])

            fin_embedding[i] = elmo_embedding

        return fin_embedding

    def dict_word2vec(self):
        """
        When create Process_data,must exec this function.
        Initial dict_word_vec.
        """
        # put the vector in the dictionary
        with open(self.output_vector_filename, "r") as f:
            i = 0
            for lines in f.readlines():
                if (i == 0):
                    i = i + 1
                    continue
                lines_split = lines.split(" ")
                keyword = lines_split[0]
                lines_split = map(float, lines_split[1:-1])
                self.dict_word_vec[keyword] = lines_split

        # Set value in "BLANK",its size is 300
        self.dict_word_vec["blank"] = np.random.normal(size=self.vector_size, loc=0, scale=0.05)

        # Set value in "<e1>","<\\e1>","<e2>","<\\e2>"
        self.dict_word_vec["<e1>"] = np.random.normal(size=self.vector_size, loc=0, scale=0.05)
        self.dict_word_vec["<\\e1>"] = np.random.normal(size=self.vector_size, loc=0, scale=0.05)
        self.dict_word_vec["<e2>"] = np.random.normal(size=self.vector_size, loc=0, scale=0.05)
        self.dict_word_vec["<\\e2>"] = np.random.normal(size=self.vector_size, loc=0, scale=0.05)

    def label2id_init(self):
        """
        When create Process_data,must exec this function.
        Change the traing label value to id.
        """
        with open(self.label2id_txt, "r") as f:
            for lines in f.readlines():
                lines = lines.strip("\r\n").split()
                self.label2id[lines[0]] = lines[1]

    def get_entity(self, dataset_txt, save_txt):
        """
        Get two entity in dataset
        """
        entity_save_txt = open(save_txt, "w")
        sen_umu = 1
        with open(dataset_txt, "r") as f:
            for lines in f.readlines():
                e1 = lines.split("<e1>")[1].split(" <\e1>")[0]
                e2 = lines.split("<e2>")[1].split(" <\e2>")[0]
                entity_save_txt.write(str(sen_umu) + " " + e1 + " <e> " + e2 + "\n")
                sen_umu += 1
        entity_save_txt.close()

if __name__ == "__main__":
    elmo_conll = ELMO_CONLL()
    start_time = time.time()

    """ 1. generate other and no other data """
    # conll04_data_no_other_list = elmo_conll.process_no_other_data()
    # conll04_data_other_list = elmo_conll.process_other_data()

    """ 2. ELMO Embedding """
    # train_data_list = elmo_conll.get_data_in_file(elmo_conll.train_sen_store_filename)
    # test_data_list = elmo_conll.get_data_in_file(elmo_conll.test_sen_store_filename)
    # conll04_data_train_elmo = elmo_conll.get_elmo_embedding(train_data_list)
    # conll04_data_test_elmo = elmo_conll.get_elmo_embedding(test_data_list)
    #
    # # save elmo embedding
    # np.save("../data/conll04/conll04_elmo/conll04_train_elmo_embedding.npy", conll04_data_train_elmo)
    # np.save("../data/conll04/conll04_elmo/conll04_test_elmo_embedding.npy", conll04_data_test_elmo)

    """ 3. load the dict word2vec """
    elmo_conll.dict_word2vec()
    elmo_conll.label2id_init()
    print("load the dict word2vec: %f s" % (time.time() - start_time))

    """ 4.load the glove embedding """
    # # traing_word_pos_vec3D:training data
    # training_word_pos_vec3D, train_pos_vec, training_sen_length, train_sen_list2D = \
    #     elmo_conll.embedding_lookup(elmo_conll.train_sen_store_filename, elmo_conll.training_e1_e2_pos_filename, elmo_conll.training_sen_number)
    # training_word_pos_vec3D = np.float32(training_word_pos_vec3D)
    # training_sen_length = np.int32(np.array(training_sen_length))
    # print("load the all glove embedding: %f s" % (time.time() - start_time))