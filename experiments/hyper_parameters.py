#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:19:46 2018

@author: llq
"""
import numpy as np
import matplotlib.pyplot as plt

y1=[87.30, 87.60, 88.10, 88.6, 87.9]
y2=[96.0,96.3,97.0, 98.14, 97.6]
y3=[73.5, 74.9, 75.0, 75.9, 74.0]
x=[100, 150, 200, 250, 300]

save_path="./gru_unit.png"
title="Performance of the number of parameter r"
plt.figure(figsize=(10,10))
#plt.title(title)
plt.grid() #open grid
plt.xlabel(r'the size $g$ of each GRU unit',fontsize=30)
plt.ylabel("F1-score(%)",fontsize=30)
plt.xticks(x, x, fontsize=30)
plt.yticks(fontsize=30)
#plt.ylim(73, 89)
plt.plot(x,y1,'o-',label="SemEval-2010 Task 8", linewidth = '4')
plt.plot(x,y2,'v-',label="CoNLL-3R", linewidth = '4')
plt.plot(x,y3,'p-',label="TAC40", linewidth = '4')
plt.legend(bbox_to_anchor=(0.6, 0.4),fontsize=25)
plt.savefig(save_path,dpi=500)
plt.show()
plt.close()


""""""
# y1=[87.8,87.9,88.30,88.4,88.6,88.5]
# y2=[96.0,96.3,97.0,97.7,98.14,97.6]
# y3=[74.2,74.4,74.8,74.9,75.9,75.1]
# x=[20,40,60,80,100,120]
#
# save_path="./the_shape_of_weight_parameter.png"
# title="Performance of the number of parameter r"
# plt.figure(figsize=(10,10))
# #plt.title(title)
# plt.grid() #open grid
# plt.xlabel('the shape $d_a$ of weight parameter $w_2^{ANA}$',fontsize=30)
# plt.ylabel("F1-score(%)",fontsize=30)
# plt.xticks(x,x,fontsize=30)
# plt.yticks(fontsize=30)
# #plt.ylim(87.0, 90.0)
# plt.plot(x,y1,'o-',label="SemEval-2010 Task 8", linewidth = '4')
# plt.plot(x,y2,'v-',label="CoNLL-3R", linewidth = '4')
# plt.plot(x,y3,'p-',label="TAC40", linewidth = '4')
# plt.legend(bbox_to_anchor=(0.6, 0.4), fontsize=25)
# plt.savefig(save_path, dpi=500)
# plt.show()
# plt.close()


""""""
# y1=[88.0,88.6,88.1,87.8,87.4]
# y2=[95.8, 98.14, 97.2, 96.6, 96.6]
# y3=[75.2,75.9,75.4,75.1,74.6]
# x=[0.005,0.01,0.015,0.02,0.025]
#
# save_path="./control_purtuation.png"
# title="Performance of the number of parameter r"
# plt.figure(figsize=(10,10))
# # plt.title(title)
# plt.grid() #open grid
# plt.xlabel(r'the hyper-parameter ' + r'$\vartheta$',fontsize=30)
# plt.ylabel("F1-score(%)",fontsize=30)
# plt.xticks(x, x, fontsize=30)
# plt.yticks(fontsize=30)
# #plt.ylim(87.0,90.0)
# plt.plot(x,y1,'o-',label="SemEval-2010 Task 8", linewidth = '4')
# plt.plot(x,y2,'v-',label="CoNLL-3R", linewidth = '4')
# plt.plot(x,y3,'p-',label="TAC40", linewidth = '4')
# plt.legend(bbox_to_anchor=(0.6, 0.4), fontsize=25)
# plt.savefig(save_path, dpi=500)
# plt.show()
# plt.close()
