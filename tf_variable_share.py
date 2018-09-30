# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 20:07:01 2018

@author: todd
"""
#import tensorflow as tf
#
#from tensorflow.python.ops import variable_scope as vs    ### 改动部分 ###
#
#def func(in_put, in_channel, out_channel, reuse=False):    ### 改动部分 ###
#
#    if reuse:                                        ### 改动部分 ###
#        vs.get_variable_scope().reuse_variables()    ### 改动部分 ###
#
#    weights = tf.get_variable(name="weights", shape=[2, 2, in_channel, out_channel],
#                              initializer=tf.contrib.layers.xavier_initializer_conv2d())
#    
#    #捲積op
#    output = tf.nn.conv2d(input = in_put, filter=weights, strides=[1, 1, 1, 1], padding="SAME")
#    return output
#
#
#def main(): 
#    with tf.Graph().as_default():
#        input_x = tf.placeholder(dtype=tf.float32, shape=[1, 4, 4, 1])
#
#        for _ in range(5):
#            output = func(input_x, 1, 1, reuse=(_!=0))    ### 改动部分 ###
#            with tf.Session() as sess:
#                sess.run(tf.global_variables_initializer())
#                import numpy as np
#                _output = sess.run(output, feed_dict={input_x:np.random.uniform(low=0, high=255, size=[1, 4, 4, 1])})
#                print (_output)
#
#if __name__ == "__main__":
#    main()


import tensorflow as tf

def func(in_put, in_channel, out_channel):
    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):    ### 改动部分 ###
        weights = tf.get_variable(name="weights", shape=[2, 2, in_channel, out_channel],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        convolution = tf.nn.conv2d(input=in_put, filter=weights, strides=[1, 1, 1, 1], padding="SAME")
    return convolution


def main():
    with tf.Graph().as_default():
        input_x = tf.placeholder(dtype=tf.float32, shape=[1, 4, 4, 1])

        for _ in range(5):
            output = func(input_x, 1, 1)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                import numpy as np
                _output = sess.run([output], feed_dict={input_x:np.random.uniform(low=0, high=255, size=[1, 4, 4, 1])})
                print (_output)

if __name__ == "__main__":
    main()
