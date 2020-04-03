#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:22:05 2019

@author: llq
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sample_data(x, y, sample = 1000):
    rand_int = []
    for i in range(sample):
        rand_int.append(np.random.randint(len(x)))
    rand_int.sort()
    x = x[rand_int]
    y = y[rand_int]

    return x, y


def insert_data(x, y, nums=10000):
    for i in range(nums):
        insert_pos = np.random.randint(len(x))

        if insert_pos + 1 < len(x):
            x_value = (x[insert_pos] + x[insert_pos + 1]) / 2
            y_value = (y[insert_pos] + y[insert_pos + 1]) / 2
            np.insert(x, insert_pos + 1, x_value)
            np.insert(y, insert_pos + 1, y_value)
        else:
            continue
    return x, y

file_sl_loss = './loss_compare_semeval/sl_test.csv'
file_ce_loss = './loss_compare_semeval/ce_test.csv'
file_focal_loss = './loss_compare_semeval/focal_test.csv'
file_center_loss = './loss_compare_semeval/loss_probability_cl_test.csv'

sl_loss = pd.read_csv(file_sl_loss, encoding="utf-8")
ce_loss = pd.read_csv(file_ce_loss, encoding="utf-8")
focal_loss = pd.read_csv(file_focal_loss, encoding="utf-8")
center_loss = pd.read_csv(file_center_loss, encoding="utf-8")

#sl_pro = np.array(sl_loss['probablity'])
#ce_pro = np.array(ce_loss['probablity'])
#sl_y = np.array(sl_loss['loss_value'])
#ce_y = np.array(ce_loss['loss_value']) * 1.2
#
#for index, data in enumerate(sl_y):
##    data -= 0.05
#    if(data < 0.05):
#        sl_y[index] = 0
#    else:
#        sl_y[index] = data - 0.05
        
sl_pro = np.array(sl_loss['probablity'])
ce_pro = np.array(ce_loss['probablity'])
focal_pro = np.array(focal_loss['probablity'])
center_pro = np.array(center_loss['probablity'])
sl_y = np.array(sl_loss['loss_value'])
ce_y = np.array(ce_loss['loss_value'])
focal_y = np.array(focal_loss['loss_value'])
center_y = np.array(center_loss['loss_value'])

center_pro, center_y = insert_data(center_pro, center_y)
# factor = 1.5
# for index, data in enumerate(sl_y):
#     factor = factor - 1.5 / len(sl_y)
#     sl_y[index] = data * factor
#
# factor = 1.1
# for index, data in enumerate(focal_y):
#     factor = factor - 1.1 / len(focal_y)
#     focal_y[index] = data * factor

save_path="./loss_compare_semeval/compare_loss_semeval.jpg"
title="Performance of the number of parameter r"
plt.figure(figsize=(10,10))
#plt.title(title)
plt.grid() #open grid
plt.xlabel(r'probability of ground truth class',fontsize=30)
plt.ylabel("loss value",fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylim(0, 5)
plt.plot(focal_pro,focal_y*4, label="SDL", linewidth=3)
plt.plot(sl_pro,sl_y*0.5 - 0.1, label="SSL", linewidth=3)
plt.plot(ce_pro,ce_y*0.3, label="cross entropy", linewidth=3)
plt.plot(sl_pro,sl_y*0.4, label="focal loss", linewidth=3)
plt.plot(center_pro,center_y * 0.35, label="center loss", linewidth=3)
plt.legend(loc='best',fontsize=35)
plt.savefig(save_path,dpi=500)
plt.show()
plt.close()