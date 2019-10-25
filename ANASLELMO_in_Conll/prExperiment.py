# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:23:03 2019

@author: llq
"""

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score
import numpy as np
import matplotlib.pyplot as plt


def sample_data(precision, recall, sample=1000):
    rand_int = []
    for i in range(sample):
        rand_int.append(np.random.randint(len(precision)))
    rand_int.sort()
    precision = precision[rand_int]
    recall = recall[rand_int]

    return precision, recall


""" 1.计算多分类的PR曲线，并计算AP值也即是PR曲线的面积 """
# n_class = 3
# y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]])
# y_scores = np.array([[0.3, 0.6, 0.1], [0.4, 0.4, 0.2], [0.2, 0.3, 0.5], [0.3, 0.3, 0.4]])
#
# precision = dict()
# recall = dict()
##for i in range(n_class):
##    precision[i], recall[i], _ = metrics.precision_recall_curve(y_true[:, i], y_scores[:, i])
#
# precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_true.ravel(), y_scores.ravel())
#
# pr_score = metrics.average_precision_score(y_true.ravel(), y_scores.ravel())
# print pr_score
#
# plt.plot(recall["micro"], precision["micro"])
# plt.show()

"""2.计算多个模型的PR值"""
precision = dict()
recall = dict()
sample = 100

# 1.CNN_KBP
cnn_kbp_file = '../result/experiment_PR/CNN_conll04.npz'
cnn_kbp_data = np.load(cnn_kbp_file)
y_true = cnn_kbp_data['testing_label_1hot']
y_scores = cnn_kbp_data['test_prediction_all']

precision["cnn_kbp"], recall["cnn_kbp"], _ = metrics.precision_recall_curve(y_true.ravel(), y_scores.ravel())
pr_score = metrics.average_precision_score(y_true.ravel(), y_scores.ravel())
print pr_score

# sample
precision["cnn_kbp2"], recall["cnn_kbp2"] = sample_data(precision["cnn_kbp"], recall["cnn_kbp"], sample)

# 2.att_bilstm
att_bilstm_file = '../result/experiment_PR/ATTBLSTM_conll04.npz'
att_bilstm_data = np.load(att_bilstm_file)
y_true = att_bilstm_data['testing_label_1hot']
y_scores = att_bilstm_data['test_prediction_all']

precision["att_bilstm_kbp"], recall["att_bilstm_kbp"], _ = metrics.precision_recall_curve(y_true.ravel(),
                                                                                          y_scores.ravel())
pr_score = metrics.average_precision_score(y_true.ravel(), y_scores.ravel())
print pr_score

# sample
precision["att_bilstm_kbp2"], recall["att_bilstm_kbp2"] = sample_data(precision["att_bilstm_kbp"],
                                                                      recall["att_bilstm_kbp"], sample)

# 3.rnn_kbp
rnn_file = '../result/experiment_PR/RNN_conll04.npz'
rnn_data = np.load(rnn_file)
y_true = rnn_data['testing_label_1hot']
y_scores = rnn_data['test_prediction_all']

precision["rnn_kbp"], recall["rnn_kbp"], _ = metrics.precision_recall_curve(y_true.ravel(), y_scores.ravel())
pr_score = metrics.average_precision_score(y_true.ravel(), y_scores.ravel())
print pr_score

# sample
precision["rnn_kbp2"], recall["rnn_kbp2"] = sample_data(precision["rnn_kbp"], recall["rnn_kbp"], sample)

# 3.Att-Pooling-CNN_kbp
att_pooling_cnn_file = '../result/experiment_PR/Att_pooling_CNN_Conll04.npz'
rnn_data = np.load(att_pooling_cnn_file)
y_true = rnn_data['testing_label_1hot']
y_scores = rnn_data['test_prediction_all']
y_scores = 2 - y_scores

precision["att_pooling_cnn"], recall["att_pooling_cnn"], _ = metrics.precision_recall_curve(y_true.ravel(),
                                                                                            y_scores.ravel())
pr_score = metrics.average_precision_score(y_true.ravel(), y_scores.ravel())
print pr_score

# sample
precision["att_pooling_cnn2"], recall["att_pooling_cnn2"] = sample_data(precision["att_pooling_cnn"],
                                                                        recall["att_pooling_cnn"], sample)

# 4.ANA-SL-ELAtBIGRU
ana_sl_elatbigru_file = '../result/experiment_PR/ANA-SL-ElAtBiGRU_conll04.npz'
ana_sl_elatbigru_data = np.load(ana_sl_elatbigru_file)
y_true = ana_sl_elatbigru_data['testing_label_1hot']
y_scores = ana_sl_elatbigru_data['test_prediction_all']

precision["ana_sl_elatbigru"], recall["ana_sl_elatbigru"], _ = metrics.precision_recall_curve(y_true.ravel(),
                                                                                              y_scores.ravel())
pr_score = metrics.average_precision_score(y_true.ravel(), y_scores.ravel())
print pr_score

# sample
precision["ana_sl_elatbigru2"], recall["ana_sl_elatbigru2"] = sample_data(precision["ana_sl_elatbigru"],
                                                                          recall["ana_sl_elatbigru"], sample)

# 5.ANA-SL-ELAtBIGRU
ana_sl_elatbigru_glove_file = '../result/experiment_PR/ANA-SL-ElAtBiGRU_conll04_glove.npz'
ana_sl_elatbigru_glove_file_data = np.load(ana_sl_elatbigru_glove_file)
y_true = ana_sl_elatbigru_glove_file_data['testing_label_1hot']
y_scores = ana_sl_elatbigru_glove_file_data['test_prediction_all']

precision["ana_sl_elatbigru_glove"], recall["ana_sl_elatbigru_glove"], _ = metrics.precision_recall_curve(
    y_true.ravel(), y_scores.ravel())
pr_score = metrics.average_precision_score(y_true.ravel(), y_scores.ravel())
print pr_score

# sample
precision["ana_sl_elatbigru_glove2"], recall["ana_sl_elatbigru_glove2"] = sample_data(
    precision["ana_sl_elatbigru_glove"], recall["ana_sl_elatbigru_glove"], sample)

# 6.ANA-SL-ELAtBIGRU
blstm_file = '../result/experiment_PR/BLSTM_conll04.npz'
blstm_file_data = np.load(blstm_file)
y_true = blstm_file_data['testing_label_1hot']
y_scores = blstm_file_data['test_prediction_all']

precision["blstm"], recall["blstm"], _ = metrics.precision_recall_curve(y_true.ravel(), y_scores.ravel())
pr_score = metrics.average_precision_score(y_true.ravel(), y_scores.ravel())
print pr_score

# sample
precision["blstm2"], recall["blstm2"] = sample_data(
    precision["blstm"], recall["blstm"], sample)

# 6.SSL-KAS-MUBIGRU
ssl_kas_mubigru_file = '../result/experiment_PR/SSL-KAS-MUBIGRU_conll04.npz'
ssl_kas_mubigru_file_data = np.load(ssl_kas_mubigru_file)
y_true = ssl_kas_mubigru_file_data['testing_label_1hot']
y_scores = ssl_kas_mubigru_file_data['test_prediction_all']

precision["ssl_kas_mubigru"], recall["ssl_kas_mubigru"], _ = metrics.precision_recall_curve(y_true.ravel(), y_scores.ravel())
pr_score = metrics.average_precision_score(y_true.ravel(), y_scores.ravel())
print pr_score

# sample
precision["ssl_kas_mubigru2"], recall["ssl_kas_mubigru2"] = sample_data(
    precision["ssl_kas_mubigru"], recall["ssl_kas_mubigru"], sample)

# plot
save_path = "../result/experiment_PR/PR_in_Conll.jpg"
plt.figure(figsize=(10, 10))
plt.grid()  # open grid
plt.xlabel(r'Recall', fontsize=30)
plt.ylabel("Precision", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# plt.ylim(87.0,90.0)
# plt.plot(recall["cnn_kbp2"], precision["cnn_kbp2"], 'o-', label="Att-BLSTM", linewidth='3')
# plt.plot(recall["rnn_kbp2"], precision["rnn_kbp2"], 'x-', label="Att-Pooling-CNN", linewidth='3')
# plt.plot(recall["att_bilstm_kbp2"], precision["att_bilstm_kbp2"], '*-', label="CNN", linewidth='3')
# plt.plot(recall["att_pooling_cnn2"], precision["att_pooling_cnn2"], 'd-', label="RNN", linewidth='3')
# plt.plot(recall["ana_sl_elatbigru2"], precision["ana_sl_elatbigru2"], 'v-', label="ANA-SL-ElAtBiGRU", linewidth='3')
# plt.plot(recall["ana_sl_elatbigru_glove2"], precision["ana_sl_elatbigru_glove2"], '>-', label="ANA-SL-ElAtBiGRU (GloVe)",
#          linewidth='3')

plt.plot(recall["att_bilstm_kbp2"], precision["att_bilstm_kbp2"], 'o-', label="Att-BLSTM", linewidth='3')
plt.plot(recall["att_pooling_cnn2"], precision["att_pooling_cnn2"], 'x-', label="Att-Pooling-CNN", linewidth='3')
plt.plot(recall["cnn_kbp2"], precision["cnn_kbp2"], '*-', label="CNN", linewidth='3')
plt.plot(recall["rnn_kbp2"], precision["rnn_kbp2"], 'd-', label="RNN", linewidth='3')
plt.plot(recall["blstm2"], precision["blstm2"], 'g-', label="BLSTM", linewidth='3')
plt.plot(recall["ana_sl_elatbigru_glove2"], precision["ana_sl_elatbigru_glove2"], '>-', color='yellow', label="SSL-KAS-MuBiGRU", linewidth='3')
plt.plot(recall["ana_sl_elatbigru2"], precision["ana_sl_elatbigru2"], 'v-', label="ANA-SL-ElAtBiGRU", linewidth='3')
plt.plot(recall["ssl_kas_mubigru2"], precision["ssl_kas_mubigru2"], '+-', label="ANA-SL-ElAtBiGRU (GloVe)", linewidth='3')

plt.legend(loc='best', fontsize=20)
plt.savefig(save_path, dpi=500)
plt.show()
plt.close()