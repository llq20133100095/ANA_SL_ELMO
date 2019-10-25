#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2019.10.25

@author: llq
@function:
    1.center loss
"""
import numpy as np
import theano.tensor as T
import theano
from theano.compile.ops import as_op

@as_op(itypes=[T.ivector, T.fmatrix],
       otypes=[T.ivector, T.ivector, T.fmatrix])
def unique_with_counts(x, sub):
    """
    将函数转换为tensor函数操作
    :param x:
    :return:
    """
    unique_y = []
    idx = []
    dict_count = {}
    count = []
    for v in x:
        if v not in unique_y:
            unique_y.append(v)
            dict_count[v] = 1
        else:
            dict_count[v] += 1

    for i in range(len(x)):
        for j in range(len(unique_y)):
            if x[i] == unique_y[j]:
                idx.append(j)

    for j in range(len(unique_y)):
        count.append(dict_count[unique_y[j]])

    count = np.array(count)
    times = count[idx]

    times = np.reshape(times, [-1, 1])
    sub = np.array(sub) / np.float32((1 + times))

    return np.array(unique_y), times, sub


@as_op(itypes=[T.fmatrix, T.ivector, T.fmatrix],
       otypes=[T.fmatrix])
def scatter_sub(ref, indices, updates):
    """
    achieve tf.scatter_sub function
    :param ref:
    :param indices:
    :param updates:
    :return:
    """
    for i in range(len(indices)):
        ref[indices[i]] -= updates[i]
    return ref


def center_loss(features, labels, alpha, centers):
    # 将特征reshape成一维
    labels = T.reshape(T.argmax(labels, axis=1), [-1])
    labels = T.cast(labels, "int32")

    # 获取当前batch每个样本对应的中心
    centers_batch = centers[labels]
    # 计算center loss的数值
    loss = T.sum((features - centers_batch) ** 2) / 2

    # 以下为更新中心的步骤
    diff = centers_batch - features

    # 获取一个batch中同一样本出现的次数，这里需要理解论文中的更新公式
    unique_label, appear_times, diff = unique_with_counts(labels, diff)
    diff = alpha * diff

    # 更新中心
    centers = scatter_sub(centers[:], labels, diff)
    return loss, centers


def neg_center_loss(features, labels, alpha, centers):
    # 将特征reshape成一维
    labels = T.reshape(T.argmax(labels, axis=1), [-1])
    labels = T.cast(labels, "int32")

    # 获取当前batch每个样本对应的中心
    centers_batch = centers[labels]
    # 计算center loss的数值
    loss = T.sum((features - centers_batch) ** 2) / 2


if __name__ == "__main__":
    features = T.fmatrix("features")
    labels = T.imatrix("labels")
    alpha = 0.5
    num_classes = 10

    centers = theano.shared(np.float32(np.zeros([num_classes, 250])), "centers")
    loss, new_centers = center_loss(features, labels, alpha, centers)
    # diff_o, appear_times3, diff3, new_centers = center_loss(features, labels, alpha, centers)

    # input value
    features_in = np.float32(np.random.uniform(size=[100, 250]))
    labels_in = np.int32(np.random.randint(low=0, high=100, size=[100, 10]))
    labels_id = np.argmax(labels_in, axis=1)
    # centers_in = np.float32(np.zeros([num_classes, 250]))

    train_fn = theano.function([features, labels], [loss, centers], updates={centers:new_centers}, on_unused_input='warn')
    # train_fn = theano.function([features, labels], [diff_o, appear_times3, diff3, new_centers], updates={centers:new_centers}, on_unused_input='warn')

    output_loss, output_centers = train_fn(features_in, labels_in)
    # output_diff_o, output_appear_times3, output_diff3, output_new_centers = train_fn(features_in, labels_in)


    """ 2.tensorflow """
    # import tensorflow as tf
    # x = tf.placeholder("float32", shape=[None], name="x")
    # y, idx, count = tf.unique_with_counts(x)
    #
    # sess = tf.InteractiveSession()
    # yy, idx_i, count_c = sess.run([y, idx, count], feed_dict={x: [1, 1, 3, 2, 3, 3, 4, 4, 4, 7, 8, 8]})


