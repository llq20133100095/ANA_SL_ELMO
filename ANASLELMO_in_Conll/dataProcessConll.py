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
# from allennlp.commands.elmo import ElmoEmbedder
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
        self.train_e1_e2_pos_filename = os.path.join(dir_path, "../data/conll04/conll04_sdp/conll04_train/train_e1_e2.txt")
        self.train_sen_number = 739

        """ 5. testing file """
        self.test_sen_store_filename = os.path.join(dir_path, "../data/conll04/conll04_relation_train_test/conll04_test_sen.txt")
        self.test_label_store_filename = os.path.join(dir_path, "../data/conll04/conll04_relation_train_test/conll04_test_label.txt")
        self.test_e1_e2_pos_filename = os.path.join(dir_path, "../data/conll04/conll04_sdp/conll04_test/test_e1_e2.txt")
        self.test_sen_number = 186

        """ 6.Position:initial the position vector """
        self.pos2vec_len = 20
        self.pos2vec_init = np.random.normal(size=(131, self.pos2vec_len), loc=0, scale=0.05)

        """ 7.Elmo save """
        self.train_elmo_file = os.path.join(dir_path, '../data/conll04/conll04_elmo/conll04_train_elmo_embedding.npy')
        self.test_elmo_file = os.path.join(dir_path,'../data/conll04/conll04_elmo/conll04_test_elmo_embedding.npy')

        """
        8.SDP file
        """
        self.e1_sdp_train_file = os.path.join(dir_path, "../data/conll04/conll04_sdp/conll04_train/train_e1_SDP.txt")
        self.e2_sdp_train_file = os.path.join(dir_path, "../data/conll04/conll04_sdp/conll04_train/train_e2_SDP.txt")
        self.e1_sdp_test_file = os.path.join(dir_path, "../data/conll04/conll04_sdp/conll04_test/test_e1_SDP.txt")
        self.e2_sdp_test_file = os.path.join(dir_path, "../data/conll04/conll04_sdp/conll04_test/test_e2_SDP.txt")

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

    """
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
    """

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

    def pos_embed(self, x):
        """
        embedding the position
        :param x:
        :return:
        """
        if x < -64:
            return 0
        if x >= -64 and x <= 64:
            return x + 65
        if x > 64:
            return 130

    def embedding_lookup(self, sen_store_filename, e1_e2_pos_filename, sen_number):
        """
        1.sen_list2D:put sentence in this.format:[[sentence1],[sentence2]]
        2.word_vec3D:get each word vector,and Make data to this format:8000*105*300.
            In 105*300,the first dim is word;the sencond dim is vector
        3.word_pos_vec3D:has "word vector" and "position vector".
            this format is N*105*320,(N has two value "8000" and "2717")
        """
        word_vec3D = np.empty((sen_number, self.max_length_sen, self.vector_size))
        #        word_pos_vec3D=np.empty((sen_number,self.max_length_sen,340))

        # sen_list:store the sentence([[sentence1],[sentence2]] )
        sen_list2D = []
        # sen_length:length of sentence
        sen_length = []

        # load the word in sen_list2D.
        # The format is:[[sentence1],[sentence2]]
        with open(sen_store_filename, "r") as f:
            sentence_id = 0
            for lines in f.readlines():

                lines = lines.replace("     ", " ").replace("    ", " ") \
                            .replace("   ", " ").replace("  ", " ").split(" ")[:-1]

                # Remove the stare " "
                if (lines[0] == ""):
                    lines = lines[1:]

                # store the original length of sentence
                sen_length.append(len(lines))
                sentence_id = sentence_id + 1

                # append the length of sen_list2D to 105 lengths.
                # And the flag is 'BLANK'
                if (len(lines) <= self.max_length_sen):
                    for i in range(self.max_length_sen - len(lines)):
                        lines.append('blank')
                sen_list2D.append(lines)

        # Find the word vector in dict_word_vec.
        # Make data to this format:N*105*300,(N has two value "8000" and "2717")
        # In 105*300,the first dim is "word";the sencond dim is "vector"
        sentence_id = 0
        for sentences in sen_list2D:
            word_id = 0
            for words in sentences:
                # find word in dict_word_vec
                if (self.dict_word_vec.has_key(words)):
                    word_vec3D[sentence_id][word_id] = self.dict_word_vec[words]
                    word_id = word_id + 1
                else:
                    self.dict_word_vec[words] = np.random.normal(size=(1, self.vector_size), loc=0, scale=0.05)
                    word_vec3D[sentence_id][word_id] = self.dict_word_vec[words]
                    word_id = word_id + 1
            #                    print "Warning: don't find word in dict_word_vec"
            sentence_id = sentence_id + 1

        pos_id = np.empty((sen_number, self.max_length_sen, 2))
        sentence_id = 0
        with open(e1_e2_pos_filename, "r") as f:
            for lines in f.readlines():
                # the two "relation word":e1,e2
                e1 = lines.split("<e>")[0].split(" ")[1:]
                e2 = lines.split("<e>")[1].strip("\n").split(" ")

                # Position number of e1 and e2
                pos_e1 = 0
                pos_e2 = 0
                # If entity word has two number and more,set this "pos_e1" and "pos_e2" are the 1st word in entity word
                for i in range(len(sen_list2D[sentence_id])):
                    if (sen_list2D[sentence_id][i] == e1[-2] and sen_list2D[sentence_id][i + 1] == "<\\e1>"):
                        pos_e1 = i

                    if (sen_list2D[sentence_id][i] == e2[-1] and sen_list2D[sentence_id][i + 1] == "<\\e2>"):
                        pos_e2 = i

                for i in range(len(sen_list2D[sentence_id])):
                    if (i == pos_e1):
                        pos_id[sentence_id][i] = \
                            np.array([self.pos_embed(0), self.pos_embed(i - pos_e2)])
                    elif (i == pos_e2):
                        pos_id[sentence_id][i] = \
                            np.array([self.pos_embed(i - pos_e1), self.pos_embed(0)])
                    else:
                        pos_id[sentence_id][i] = \
                            np.array([self.pos_embed(i - pos_e1), self.pos_embed(i - pos_e2)])
                sentence_id = sentence_id + 1

        # Set the "position word" to vector.
        # pos_vec:N(sentence)*105(word)*20(position vector),(N has two value "8000" and "2717")
        pos_vec = np.empty((sen_number, self.max_length_sen, 40))
        sentence_id = 0
        for word in pos_id:
            i = 0
            for pos_num in word:
                pos_vec[sentence_id][i] = np.hstack \
                    ((self.pos2vec_init[int(pos_num[0])], self.pos2vec_init[int(pos_num[1])]))
                i = i + 1
            sentence_id = sentence_id + 1

        return word_vec3D, pos_vec, sen_length, sen_list2D

    def merge_glove_elmo(self, word_pos_vec3D, pos_vec, elmo_file):
        """
        Function:
            1.merge the word_pos_vec3D and elmo_embedding
            2.word_pos_vec3D: [glove embedding, position embedding]
        Parameter:
            1.word_pos_vec3D: embedding
            2.elmo_file: save the ELMO embedding
        """
        elmo_embedding=np.load(elmo_file)
        word_vec3D = np.concatenate((word_pos_vec3D, elmo_embedding, pos_vec), axis=2)
        return word_vec3D

    def label2id_in_data(self, label_store_filename, sen_number):
        """
        In train or test data,change the traing label value to id.
        """
        data_label = np.empty((sen_number)).astype(int)

        label_number = 0
        with open(label_store_filename, "r") as f:
            for lines in f.readlines():
                data_label[label_number] = self.label2id[lines.strip("\r\n")]
                label_number = label_number + 1

        return data_label

    def embedding_looking_root_e1_e2(self, e1_sdp_file, e2_sdp_file, sen_number, sen_list2D, elmo_file):
        """
        Function:
            embedding the "root" and e1 and e2
        """
        # store the root word
        root_list = []
        # store the e1 word
        e1_list = []
        with open(e1_sdp_file, "r") as f:
            for lines in f.readlines():
                root = lines.split(" ")[0].replace("'", "")
                # get the format such as "book-crossing"
                if "-" in root:
                    root = root.split("-")[1]

                e1 = lines.strip("\r\n").split(" ")[-2]
                root_list.append(root)
                e1_list.append(e1)

        # store the e2 word
        e2_list = []
        with open(e2_sdp_file, "r") as f:
            for lines in f.readlines():
                e2 = lines.strip("\r\n").split(" ")[-2]
                e2_list.append(e2)

        # load the elmo_embedding
        elmo_embedding = np.load(elmo_file)

        # root embedding and elmo_embedding
        root_embedding = np.zeros((sen_number, self.vector_size + 1024))
        sen_num = 0
        for root in root_list:
            try:
                index = sen_list2D[sen_num].index(root)
                elmo = elmo_embedding[sen_num][index]
            except:
                elmo = np.random.normal(size=(1024,), loc=0, scale=0.05)

            try:
                root_embedding[sen_num] = np.concatenate((self.dict_word_vec[root], elmo), axis=0)
            except:
                self.dict_word_vec[root] = np.random.normal(size=(self.vector_size,), loc=0, scale=0.05)
                root_embedding[sen_num] = np.concatenate((self.dict_word_vec[root], elmo), axis=0)
            sen_num += 1

        # e1 embedding
        e1_embedding = np.zeros((sen_number, self.vector_size + 1024))
        sen_num = 0
        for e1 in e1_list:
            try:
                index = sen_list2D[sen_num].index(e1)
                elmo = elmo_embedding[sen_num][index]
            except:
                elmo = np.random.normal(size=(1024,), loc=0, scale=0.05)

            try:
                e1_embedding[sen_num] = np.concatenate((self.dict_word_vec[e1], elmo), axis=0)
            except:
                self.dict_word_vec[e1] = np.random.normal(size=(self.vector_size,), loc=0, scale=0.05)
                e1_embedding[sen_num] = np.concatenate((self.dict_word_vec[e1], elmo), axis=0)
            sen_num += 1

        # e2 embedding
        e2_embedding = np.zeros((sen_number, self.vector_size + 1024))
        sen_num = 0
        for e2 in e2_list:
            try:
                index = sen_list2D[sen_num].index(e2)
                elmo = elmo_embedding[sen_num][index]
            except:
                elmo = np.random.normal(size=(1024,), loc=0, scale=0.05)

            try:
                e2_embedding[sen_num] = np.concatenate((self.dict_word_vec[e2], elmo), axis=0)
            except:
                self.dict_word_vec[e2] = np.random.normal(size=(self.vector_size,), loc=0, scale=0.05)
                e2_embedding[sen_num] = np.concatenate((self.dict_word_vec[e2], elmo), axis=0)
            sen_num += 1

        # set position embedding in root,e1 and e2
        root_pos_emb = np.zeros((sen_number, self.pos2vec_len * 2))
        e1_pos_emb = np.zeros((sen_number, self.pos2vec_len * 2))
        e2_pos_emb = np.zeros((sen_number, self.pos2vec_len * 2))

        for sentence_id in range(len(sen_list2D)):
            # Position number of root, e1 and e2
            pos_root = 0
            pos_e1 = 0
            pos_e2 = 0
            # If entity word has two number and more,set this "pos_e1" and "pos_e2" are the 1st word in entity word
            for i in range(len(sen_list2D[sentence_id])):
                if (sen_list2D[sentence_id][i] == root_list[sentence_id]):
                    pos_root = i

                if (sen_list2D[sentence_id][i] == e1_list[sentence_id] and sen_list2D[sentence_id][
                    i + 1] == "<\\e1>"):
                    pos_e1 = i

                if (sen_list2D[sentence_id][i] == e2_list[sentence_id] and sen_list2D[sentence_id][
                    i + 1] == "<\\e2>"):
                    pos_e2 = i

            root_pos_emb[sentence_id] = np.hstack \
                ((self.pos2vec_init[int(self.pos_embed(pos_root - pos_e1))],
                  self.pos2vec_init[int(self.pos_embed(pos_root - pos_e2))]))
            e1_pos_emb[sentence_id] = np.hstack \
                ((self.pos2vec_init[int(self.pos_embed(0))],
                  self.pos2vec_init[int(self.pos_embed(pos_e1 - pos_e2))]))
            e2_pos_emb[sentence_id] = np.hstack \
                ((self.pos2vec_init[int(self.pos_embed(pos_e2 - pos_e1))],
                  self.pos2vec_init[int(self.pos_embed(0))]))

        # concate word embedding and pos embedding
        root_embedding = np.concatenate((root_embedding, root_pos_emb), axis=1)
        e1_embedding = np.concatenate((e1_embedding, e1_pos_emb), axis=1)
        e2_embedding = np.concatenate((e2_embedding, e2_pos_emb), axis=1)

        return np.float32(root_embedding), np.float32(e1_embedding), np.float32(e2_embedding)

    def label2id_1hot(self,data_label,label2id):
        """
        Make the label in one-hot encode:[0,0,...,0,1,0,0,...,0]
        """
        onehot_encoded=[]
        for value in data_label:
            onehot=np.zeros((len(label2id)))
            onehot[value]=1
            onehot_encoded.append(onehot)
        return np.array(onehot_encoded)

    def mask_train_input(self, y_train, num_labels='all'):
        """
        2.Mask_train:mask train label.When mask=1,label can be supervised.When mask=0,label can be unsupervised.
        """
        # Construct mask_train. It has a zero where label is unknown, and one where label is known.
        if num_labels == 'all':
            # All labels are used.
            mask_train = np.ones(len(y_train), dtype=np.float32)
            print("Keeping all labels.")
        else:
            #Rough classification
            rou_num_classes=10
            # Assign labels to a subset of inputs.
            max_count = num_labels // rou_num_classes
            print("Keeping %d labels per rough class." % max_count)
            mask_train = np.zeros(len(y_train), dtype=np.float32)
            count = [0] * rou_num_classes
            for i in range(len(y_train)):
                label = y_train[i]
                rou_label=int(label)/2
                if (count[rou_label]) < max_count:
                    mask_train[i] = 1.0
                count[rou_label] += 1

        return mask_train

    def iterate_minibatches_inputAttRootE1E2(self, inputs, targets, sen_length, batchsize, input_root, input_e1, input_e2, shuffle=False):
        """
        Get minibatches in input attention
        """
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt], sen_length[excerpt], input_root[excerpt], input_e1[excerpt], input_e2[excerpt]

    def iterate_minibatches(self, inputs, targets, sen_length, batchsize, shuffle=False):
        """
        Get minibatches
        """
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt], sen_length[excerpt]

if __name__ == "__main__":
    elmo_conll = ELMO_CONLL()
    start_time = time.time()

    """ 1. generate other and no other data"""
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

    """ 3. generate entity filename"""
    # elmo_conll.get_entity(elmo_conll.train_sen_store_filename, elmo_conll.train_e1_e2_pos_filename)
    # elmo_conll.get_entity(elmo_conll.test_sen_store_filename, elmo_conll.test_e1_e2_pos_filename)

    """ 4. load the dict word2vec """
    elmo_conll.dict_word2vec()
    elmo_conll.label2id_init()
    print("load the dict word2vec: %f s" % (time.time() - start_time))

    """ 5.load the glove embedding """
    # traing_word_pos_vec3D: training data
    train_word_pos_vec3D, train_pos_vec, train_sen_length, train_sen_list2D = \
        elmo_conll.embedding_lookup(elmo_conll.train_sen_store_filename, elmo_conll.train_e1_e2_pos_filename, elmo_conll.train_sen_number)
    train_word_pos_vec3D = np.float32(train_word_pos_vec3D)
    train_sen_length = np.int32(np.array(train_sen_length))
    print("load the train glove embedding: %f s" % (time.time() - start_time))

    # testing_word_pos_vec3D: testing data
    test_word_pos_vec3D, test_pos_vec, test_sen_length, test_sen_list2D = \
        elmo_conll.embedding_lookup(elmo_conll.test_sen_store_filename, elmo_conll.test_e1_e2_pos_filename, elmo_conll.test_sen_number)
    test_word_pos_vec3D = np.float32(test_word_pos_vec3D)
    test_sen_length = np.int32(np.array(test_sen_length))
    print("load the test glove embedding: %f s" % (time.time() - start_time))

    """ 6.merge the all embedding """
    train_word_pos_vec3D = elmo_conll.merge_glove_elmo(train_word_pos_vec3D, train_pos_vec, elmo_conll.train_elmo_file)
    del train_pos_vec
    test_word_pos_vec3D = elmo_conll.merge_glove_elmo(test_word_pos_vec3D, test_pos_vec, elmo_conll.test_elmo_file)
    del test_pos_vec
    print("merge the all embedding: %f s" % (time.time() - start_time))

    """ 7.load the label """
    # 4.training label
    train_label = elmo_conll.label2id_in_data(elmo_conll.train_label_store_filename, elmo_conll.train_sen_number)
    train_label = np.int32(train_label)

    # 5.testing label
    test_label = elmo_conll.label2id_in_data(elmo_conll.test_label_store_filename, elmo_conll.test_sen_number)
    test_label = np.int32(test_label)

    """ 8.load the embedding of root, e1 and e2. """
    train_root_embedding, train_e1_embedding, train_e2_embedding = \
        elmo_conll.embedding_looking_root_e1_e2(elmo_conll.e1_sdp_train_file, elmo_conll.e2_sdp_train_file, elmo_conll.train_sen_number, train_sen_list2D, elmo_conll.train_elmo_file)

    test_root_embedding, test_e1_embedding, test_e2_embedding=\
        elmo_conll.embedding_looking_root_e1_e2(elmo_conll.e1_sdp_test_file, elmo_conll.e2_sdp_test_file, elmo_conll.test_sen_number, test_sen_list2D, elmo_conll.test_elmo_file)
    print("load the embedding of root, e1 and e2: %f s" % (time.time() - start_time))

    """ 8.label id value and one-hot """
    label2id = elmo_conll.label2id
    train_label_1hot = elmo_conll.label2id_1hot(train_label, label2id)
    train_label_1hot = np.int32(train_label_1hot)

    test_label_1hot = elmo_conll.label2id_1hot(test_label, label2id)
    test_label_1hot = np.int32(test_label_1hot)
    del train_label
    del test_label