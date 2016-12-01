import sys

import tensorflow as tf

import cv2 as cv

import time

import random as rnd

import numpy as np

import Input_Data_Handler as IDH


###################################################
#TensorFlow Function Wrapper#######################
###################################################
def weight_variable(shape):
#creates a TensorFlow variable with given shape and a standard deviation 0.1
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
# creates a TensorFlow variable with given shape and a fixed value of 0.1
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
    #https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#conv2d
    #https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#convolution
    #Computes a 2-D convolution given 4-D input and filter tensors.
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main():

    train = False

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

    #A TensorFlow Session for use in interactive contexts, such as a shell.
    #https://www.tensorflow.org/versions/r0.11/api_docs/python/client.html#InteractiveSession
    sess = tf.InteractiveSession()

    #Placeholder
    #TensorFlow provides a placeholder operation that must be fed with data on execution.
    #https://www.tensorflow.org/versions/r0.11/api_docs/python/io_ops.html#placeholders

    #tf.placeholder(dtype, shape=None, name=None)
    #dtype: The type of elements in the tensor to be fed.
    #shape: The shape of the tensor to be fed (optional). If the shape is not specified, you can feed a tensor of any shape.
    #name: A name for the operation (optional).

    #Placeholder for input batch data shape=[batch_size, NrOfPixel]
    x = tf.placeholder(tf.float32, shape=[None, 50,50])

    #Placeholder for input labels shape=[batch_size, NrOfClasses]
    #Each label is a one hot vector: (1,0,0,0) ... (0,0,0,1)
    y_ = tf.placeholder(tf.float32, shape=[None, 4])

    #Variables
    #https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#Variable
    #Varaibles are Tensors with a fixed shape and initial values

    #tf.Variable(<initial-value>, name=<optional-name>)

    W = tf.Variable(tf.zeros([2500,4]))
    b = tf.Variable(tf.zeros([4]))

    #shape for all weight variables for convolution layer 1
    feature_size_x_conv1 = 5
    feature_size_y_conv1 = 5
    nrOfInputChannels_conv1 = 1
    nrOfOutputChannels_conv1 = 32       #Nr of features

    #weight_variable(shape):
    W_conv1 = weight_variable([feature_size_x_conv1, feature_size_y_conv1, nrOfInputChannels_conv1, nrOfOutputChannels_conv1])

    #bias_variable(shape):
    #creates a TensorFlow variable with
    #shape = nrOfOutputChannels_conv1
    #and a fixed value of 0.1
    b_conv1 = bias_variable([nrOfOutputChannels_conv1])

    image_width = 50
    image_height = 50
    nrOfColourChannels = 1
    x_image = tf.reshape(x, [-1, image_width, image_height, nrOfColourChannels])

    #tf.nn.relu(features, name=None)
    #https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#relu       h
    #Computes rectified linear: max(features, 0).

    #Args:
    #features: A Tensor. Must be one of the following types: float32, float64, int32, int64, uint8, int16, int8, uint16, half.
    #name: A name for the operation (optional).

    #Returns:
    #A Tensor. Has the same type as features

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

    #densly (fully) connected layer
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

    ###################################################
    # TensorFlow Saver ################################
    ###################################################

    # standard implementation
    saver = tf.train.Saver(max_to_keep=2)

    latest_checkpoint = tf.train.latest_checkpoint("SavedAmpelPhasenCNN")

    # define a global_step for restore process
    training_step = 0

    if latest_checkpoint != None:
        print("Letzter Speicherpunkt: " + str(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
        subStrings = latest_checkpoint.split("-")
        training_step = np.int(subStrings[1]) + 1

    ###################################################
    # TensorFlow SummeryWriter#########################
    ###################################################

    #creates a summary of the whole graph and saves it into folder "SavedGraphDefinitions"
    summary_writer = tf.train.SummaryWriter('SavedGraphDefinitions', sess.graph)

    ###################################################
    # TensorFlow Run Graph with feed_dict##############
    ###################################################
    if train == True:
        for i in range(training_step,20000):
          batch , labels , errors = myhandler.get_batch(100)

          #print("Batchsize: {0}".format(str(len(batch))))
          print("Step: %d" % i)
          if i%100 == 0:
              train_accuracy = accuracy.eval(feed_dict={
                  x: batch, y_: labels, keep_prob: 1.0})
              print("step %d, training accuracy %g" % (i, train_accuracy))
              saver.save(sess, 'SavedAmpelPhasenCNN/AmpelPhasenCNN', global_step=i)
          train_step.run(feed_dict={x: batch, y_: labels, keep_prob: 0.5})
        print("---------FINISHED--------")
    else:
        #Evaluation HERE
        batch1, labels1, errors1 = myhandler.get_batch(1)
        #feed_dict = {x: [batch1]}
        classification = y_conv.eval(feed_dict={x: batch1,keep_prob:1.0})
        print("classification"+str(classification))
        print("correct label" + str(labels1))

if __name__ == "__main__":
    main()