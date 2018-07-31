#!/usr/bin/env python
#-*- coding:utf-8 -*-

import tensorflow as tf

checkpoint = './log/fully_connected_feed/'
graph_path = './log/fully_connected_feed/model.ckpt-1999.meta'
sess_path = './log/fully_connected_feed/model.ckpt'
saver = tf.train.import_meta_graph(graph_path)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint))

print('success!')

