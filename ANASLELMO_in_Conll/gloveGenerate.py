#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2019.9.10 15:12

@author: llq
"""
data = []


def read_train_test(data):
    """
    Get train and test word.
    """
    # load train_sen.txt
    with open("../data/conll04/conll04_relation/conll04_other_sen.txt") as f:
        for lines in f.readlines():
            lines = lines.strip("\n").split()
            for word in lines:
                data.append(word)

    # load train_sen.txt
    with open("../data/conll04/conll04_relation/conll04_no_other_sen.txt") as f:
        for lines in f.readlines():
            lines = lines.strip("\n").split()
            for word in lines:
                data.append(word)

    return data


# make data have "no repetition" word
data_set = set(read_train_test(data))


def file_bigdata_read(filepath):
    """
    Read each lines
    """
    # look the "GoogleNews.txt" and filter the included key
    google_file = open(filepath)
    while True:
        # each read one lines
        lines = google_file.readline()
        if not lines:
            break
        yield lines


filepath = "../data/glove.6B.300d.txt"
vector_save = open("../data/conll04/conll04_relation_glove/conll04_glove_300.txt", "w")
# search in "GoogleNews.txt" and data_set,get included word.
word_num = 0
for lines in file_bigdata_read(filepath):
    lines = lines.split()
    for word in data_set:
        if lines[0].lower() == word:
            word_num += 1
            vector_save.write(lines[0] + " ")
            for data in lines[1:]:
                vector_save.write(data + " ")
            vector_save.write("\n")

print(word_num)
vector_save.close()

# write in start file.
with open("../data/conll04/conll04_relation_glove/conll04_glove_300.txt", "r+") as f:
     old = f.read()
     f.seek(0)
     f.write(str(word_num)+" "+str(300)+"\n")
     f.write(old)

