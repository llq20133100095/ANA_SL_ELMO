#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:19:46 2018

@author: llq
"""
import numpy as np
import matplotlib.pyplot as plt

y1=[96.30, 96.60, 97.10, 98.1, 97.9]
y2=[73.5, 74.9, 75.0, 75.9, 74.0]
x=[100, 150, 200, 250, 300]

save_path="../result/hyper_parameter_pic/gru_unit.jpg"
title="Performance of the number of parameter r"
plt.figure(figsize=(10,10))
#plt.title(title)
plt.grid() #open grid
plt.xlabel(r'the size $g$ of each GRU unit',fontsize=30)
plt.ylabel("F1-score(%)",fontsize=30)
plt.xticks(x, x, fontsize=30)
plt.yticks(fontsize=30)
#plt.ylim(73, 89)
plt.plot(x,y1,'o-',label="CoNLL-3R", linewidth = '4')
plt.plot(x,y2,'v-',label="TAC40", linewidth = '4')
plt.legend(loc='best',fontsize=30)
plt.savefig(save_path,dpi=500)
plt.show()
plt.close()


""""""
# y1=[96.8,96.9,97.30,97.4,98.1,97.6]
# y2=[74.2,74.4,74.8,74.9,75.9,75.1]
# x=[20,40,60,80,100,120]
#
# save_path="../result/hyper_parameter_pic/the_shape_of_weight_parameter.jpg"
# title="Performance of the number of parameter r"
# plt.figure(figsize=(10,10))
# #plt.title(title)
# plt.grid() #open grid
# plt.xlabel('the shape $d_a$ of weight parameter $w_2^{ANA}$',fontsize=30)
# plt.ylabel("F1-score(%)",fontsize=30)
# plt.xticks(x,x,fontsize=30)
# plt.yticks(fontsize=30)
# #plt.ylim(87.0, 90.0)
# plt.plot(x,y1,'o-',label="CoNLL-3R", linewidth = '4')
# plt.plot(x,y2,'v-',label="TAC40", linewidth = '4')
# plt.legend(loc='best',fontsize=30)
# plt.savefig(save_path,dpi=500)
# plt.show()
# plt.close()


""""""
# y1=[96.0,98.1,97.5,96.2,96.0]
# y2=[75.2,75.9,75.4,75.1,74.6]
# x=[0.005,0.01,0.015,0.02,0.025]
#
# save_path="../result/hyper_parameter_pic/control_purtuation.jpg"
# title="Performance of the number of parameter r"
# plt.figure(figsize=(10,10))
# #plt.title(title)
# plt.grid() #open grid
# plt.xlabel(r'the hyper-parameter ' + r'$\vartheta$',fontsize=30)
# plt.ylabel("F1-score(%)",fontsize=30)
# plt.xticks(x, x, fontsize=30)
# plt.yticks(fontsize=30)
# #plt.ylim(87.0,90.0)
# plt.plot(x,y1,'o-',label="CoNLL-3R", linewidth = '4')
# plt.plot(x,y2,'v-',label="TAC40", linewidth = '4')
# plt.legend(loc='best',fontsize=30)
# plt.savefig(save_path,dpi=500)
# plt.show()
# plt.close()
