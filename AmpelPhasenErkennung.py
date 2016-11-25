import sys

import tensorflow as tf

import cv2 as cv

import time

import random as rnd

import Input_Data_Handler as IDH


###################################################
#TensorFlow Function Wrapper#######################
###################################################
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main():
    ###################################################
    # Input_Data_Handler###############################
    ###################################################

    #initialize Input_Data_Handler
    myhandler = IDH.InputDataHandler()
    #read images into Input_Data_Handler
    myhandler.get_images_from_files()

    ###################################################
    #TensorFlow Graph definition#######################
    ###################################################

    sess = tf.InteractiveSession()

    #x = tf.placeholder(tf.float32, shape=[None, 2500])
    x = tf.placeholder(tf.float32)
    #x = tf.placeholder(tf.float32, shape=[None, 50, 50, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 4])

    W = tf.Variable(tf.zeros([None,4]))
    b = tf.Variable(tf.zeros([4]))


    feature_size_x_conv1 = 5
    feature_size_y_conv1 = 5
    nrOfInputChannels_conv1 = 1
    nrOfOutputChannels_conv1 = 32       #Nr of features
    W_conv1 = weight_variable([feature_size_x_conv1, feature_size_y_conv1, nrOfInputChannels_conv1, nrOfOutputChannels_conv1])
    b_conv1 = bias_variable([nrOfOutputChannels_conv1])

    image_width = 50
    image_height = 50
    nrOfColourChannels = 1
    x_image = tf.reshape(x, [-1, image_width, image_height, nrOfColourChannels])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #second layer
    feature_size_x_conv2 = 5
    feature_size_y_conv2 = 5
    nrOfInputChannels_conv2 = 32
    nrOfOutputChannels_conv2 = 64  # Nr of features
    W_conv2 = weight_variable([feature_size_x_conv2, feature_size_y_conv2, nrOfInputChannels_conv2, nrOfOutputChannels_conv2])
    b_conv2 = bias_variable([nrOfOutputChannels_conv2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #densly connected layer
    W_fc1 = weight_variable([13 * 13 * 64, 2048])
    b_fc1 = bias_variable([2048])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 13 * 13 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #read out layer
    W_fc2 = weight_variable([2048, 4])
    b_fc2 = bias_variable([4])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.initialize_all_variables())

    summary_writer = tf.train.SummaryWriter('mylog.log', sess.graph)

    ###################################################
    # TensorFlow Run Graph with feed_dict##############
    ###################################################
    for i in range(20000):
      batch , labels , errors = myhandler.get_batch(50)

      print("Batchsize: {0}".format(str(len(batch))))
      if i%100 == 0:
          train_accuracy = accuracy.eval(feed_dict={
              x: batch, y_: labels, keep_prob: 1.0})
          print("step %d, training accuracy %g" % (i, train_accuracy))
      train_step.run(feed_dict={x_image: batch, y_: labels, keep_prob: 0.5})

      print("End")
      #print("test accuracy %g"%accuracy.eval(feed_dict={
    #    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == "__main__":
    main()