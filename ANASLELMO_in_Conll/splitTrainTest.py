#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2019.9.21 19:17

@author: llq
@function:
    1.split training data and testing data
"""
import os
from sklearn.model_selection import StratifiedKFold
import numpy as np


def save_file(file, data):
    for line in data:
        file.write(line)


def split_data(data_sen_file, data_label_file, save_path):
    data_sen_all = []
    with open(data_sen_file, "r") as f:
        while True:
            line = f.readline()
            if line:
                data_sen_all.append(line)
            else:
                break

    data_label_all = []
    with open(data_label_file, "r") as f:
        while True:
            line = f.readline()
            if line:
                data_label_all.append(line)
            else:
                break

    #split train_data: train and val:42
    random_seed = 2019
    skf = StratifiedKFold(n_splits=5, random_state=random_seed, shuffle=True)
    for (train_in, test_in) in skf.split(data_sen_all, data_label_all):
        x_train, y_train, x_test, y_test = np.array(data_sen_all)[train_in], np.array(data_label_all)[train_in], np.array(data_sen_all)[test_in], np.array(data_label_all)[test_in]
        break

    train_sen_file = open(os.path.join(save_path, "conll04_train_sen.txt"), "w")
    train_label_file = open(os.path.join(save_path, "conll04_train_label.txt"), "w")
    test_sen_file = open(os.path.join(save_path, "conll04_test_sen.txt"), "w")
    test_label_file = open(os.path.join(save_path, "conll04_test_label.txt"), "w")

    save_file(train_sen_file, x_train)
    save_file(train_label_file, y_train)
    save_file(test_sen_file, x_test)
    save_file(test_label_file, y_test)


    train_sen_file.close()
    train_label_file.close()
    test_sen_file.close()
    test_label_file.close()
    return


if __name__ == "__main__":
    dir_path = os.path.dirname(__file__)
    data_sen_file = os.path.join(dir_path, "../data/conll04/conll04_relation/conll04_all_sen.txt")
    data_label_file = os.path.join(dir_path, "../data/conll04/conll04_relation/conll04_all_sen_label.txt")
    save_path = os.path.join(dir_path, "../data/conll04/conll04_relation_train_test/")
    split_data(data_sen_file, data_label_file, save_path)



