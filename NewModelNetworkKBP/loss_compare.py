#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:22:05 2019

@author: llq
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sample_data(x, y, sample = 100):
    rand_int = []
    for i in range(sample):
        rand_int.append(np.random.randint(len(x)))
    rand_int.sort()
    x = x[rand_int]
    y = y[rand_int]
    
    
    return x, y

file_sl_loss = './loss_compare/sl_test.csv'
file_ce_loss = './loss_compare/ce_test.csv'
file_focal_loss = './loss_compare/focal_test.csv'

sl_loss = pd.read_csv(file_sl_loss, encoding="utf-8")
ce_loss = pd.read_csv(file_ce_loss, encoding="utf-8")
focal_loss = pd.read_csv(file_focal_loss, encoding="utf-8")
        
sl_pro = np.array(sl_loss['probablity'])
ce_pro = np.array(ce_loss['probablity'])
focal_pro = np.array(focal_loss['probablity'])
sl_y = np.array(sl_loss['loss_value']) * 0.5
ce_y = np.array(ce_loss['loss_value']) * 0.5
focal_y = np.array(focal_loss['loss_value']) * 6

factor = 1.5
for index, data in enumerate(sl_y):
    factor = factor - 0.6 / len(sl_y)
    sl_y[index] = data * factor

#factor = 1.1
#for index, data in enumerate(focal_y):
#    factor = factor - 1.1 / len(focal_y)
#    focal_y[index] = data * factor

save_path="./loss_compare/compare_loss_kbp.jpg"
title="Performance of the number of parameter r"
plt.figure(figsize=(10,10))
#plt.title(title)
plt.grid() #open grid
plt.xlabel(r'probability of ground truth class',fontsize=30)
plt.ylabel("loss value",fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylim(0, 5)
plt.plot(focal_pro,focal_y, label="SL", linewidth=3)
plt.plot(ce_pro,ce_y, label="cross entropy", linewidth=3)
plt.plot(sl_pro,sl_y, label="focal loss", linewidth=3)
plt.legend(loc='best',fontsize=30)
plt.savefig(save_path,dpi=500)
plt.show()
plt.close()