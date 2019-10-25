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

@as_op(itypes=[T.ivector],
       otypes=[T.ivector, T.ivector, T.ivector])
def unique_with_counts(x):
    """
    将函数转换为tensor函数操作
    :param x:
    :return:
    """
    y = []
    idx = []
    dict_count = {}
    count = []
    for v in x:
        if v not in y:
            y.append(v)
            dict_count[v] = 1
        else:
            dict_count[v] += 1

    for i in range(len(x)):
        for j in range(len(y)):
            if x[i] == y[j]:
                idx.append(j)

    for j in range(len(y)):
        count.append(dict_count[y[j]])

    return np.array(y), np.array(idx), np.array(count)


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
    unique_label, unique_idx, unique_count = unique_with_counts(labels)
    appear_times = unique_count[unique_idx]
    appear_times = T.reshape(appear_times, [-1, 1])

    diff = diff / T.cast((1 + appear_times), "float32")
    diff = alpha * diff

    # 更新中心
    centers = scatter_sub(centers, labels, diff)

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
    labels = T.fmatrix("labels")
    alpha = 0.5
    num_classes = 10

    centers = theano.shared(np.float32(np.zeros([num_classes, 250])), "centers")
    loss, new_centers = center_loss(features, labels, alpha, centers)

    # input value
    features_in = np.float32(np.random.uniform(size=[100, 250]))
    labels_in = np.float32(np.random.randint(low=0, high=100, size=[100, 10]) / 100)
    labels_id = np.argmax(labels_in, axis=1)
    # centers_in = np.float32(np.zeros([num_classes, 250]))

    train_fn = theano.function([features, labels], [loss, new_centers], on_unused_input='warn')

    output_loss, output_centers = train_fn(features_in, labels_in)


    """ 2.tensorflow """
    # import tensorflow as tf
    # x = tf.placeholder("float32", shape=[None], name="x")
    # y, idx, count = tf.unique_with_counts(x)
    #
    # sess = tf.InteractiveSession()
    # yy, idx_i, count_c = sess.run([y, idx, count], feed_dict={x: [1, 1, 3, 2, 3, 3, 4, 4, 4, 7, 8, 8]})


