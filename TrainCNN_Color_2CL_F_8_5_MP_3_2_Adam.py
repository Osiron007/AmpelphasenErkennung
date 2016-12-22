import sys

import tensorflow as tf

import cv2 as cv

import time

import random as rnd

import numpy as np

import Input_Data_Handler as IDH

def main():



    ###################################################
    # Input_Data_Handler###############################
    ###################################################

    #initialize Input_Data_Handler
    myhandler = IDH.InputDataHandler()
    #read images into Input_Data_Handler
    #myhandler.get_images_from_files()


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

    #Placeholder for input batch data shape=[batch_size, NrOfPixel height, NrOfPixel weight, color channels]
    x = tf.placeholder(tf.float32, shape=[None, 50,50,3])

    #Placeholder for input labels shape=[batch_size, NrOfClasses]
    #Each label is a one hot vector: (1,0,0,0) ... (0,0,0,1)
    y_ = tf.placeholder(tf.float32, shape=[None, 4])

    image_width = 50
    image_height = 50
    nrOfColourChannels = 3
    x_image = tf.reshape(x, [-1, image_width, image_height, nrOfColourChannels])

    ###############First Convolution#################
    filter_height_conv1 = 5
    filter_width_conv1 = 5
    number_of_filters = 8
    number_of_channels = 3
    #to learn our features we need a variable for each pixel in each filter which will optimized during training
    filter_conv1 = tf.Variable(tf.truncated_normal([filter_height_conv1,filter_width_conv1,number_of_channels,number_of_filters], stddev=0.1))

    #for each filter we need a bias variable which can be optimizied during training
    filter_bias_conv1 = tf.Variable(tf.constant(0.1, shape=[number_of_filters*number_of_channels]))

##### Convolution with tf.nn.conv2d ######################################################################
    #with this convolution layer we want to convolute out input image with a number of filters
    #the output will be multiple pictures, one for each filter

    # input tensor of shape [batch, in_height, in_width, in_channels]
    # filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    # strides: A list of ints. 1-D of length 4.
    #          The stride of the sliding window for each dimension of input.
    #          Must be in the same order as the dimension specified with format.
    # padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
    # Returns: A Tensor. Has the same type as input
    # -> conv1 = tf.nn.conv2d(x_image, filter_conv1, strides=[1, 1, 1, 1], padding='SAME')
###########################################################################################################

    #######ACHTUNG NEUER Convolution Alg##################

##### Convolution with tf.nn.depthwise_conv2d #############################################################
    #tf.nn.depthwise_conv2d(input, filter, strides, padding, name=None)
    #Args:
    #input: 4-D with shape [batch, in_height, in_width, in_channels].
    #filter: 4-D with shape [filter_height, filter_width, in_channels, channel_multiplier].
    #strides: 1-D of size 4. The stride of the sliding window for each dimension of input.
    #padding: A string, either 'VALID' or 'SAME'. The padding algorithm. See the comment here
    #name: A name for this operation (optional).

    #Returns:
    #A 4D Tensor of shape[batch, out_height, out_width, in_channels * channel_multiplier].

    conv1 = tf.nn.depthwise_conv2d(x_image, filter_conv1, strides=[1, 1, 1, 1], padding='SAME')
###########################################################################################################

    #to avoid negative numbers in our matrices we set each value < 0 to 0 with the relu operation
    relu_conv1 = tf.nn.relu(conv1 + filter_bias_conv1)

    #to reduce the pixels in our image we maxpool

    maxpool_conv1 = tf.nn.max_pool(relu_conv1, ksize=[1, 3, 3, 1],strides=[1, 3, 3, 1], padding='SAME')


    ###############Second Convolution#################
    filter_height_conv2 = 3
    filter_width_conv2 = 3
    number_of_filters_conv2 = 5
    number_of_channels_conv2 = 24 #3*8
    # to learn our features we need a variable for each pixel in each filter which will optimized during training
    filter_conv2 = tf.Variable(
        tf.truncated_normal([filter_height_conv2, filter_width_conv2, number_of_channels_conv2, number_of_filters_conv2],
                            stddev=0.1))

    # for each filter we need a bias variable which can be optimizied during training
    filter_bias_conv2 = tf.Variable(tf.constant(0.1, shape=[number_of_filters_conv2*number_of_channels_conv2]))

    # with this convolution layer we want to convolute out input image with a number of filters
    # the output will be multiple pictures, one for each filter
    #conv2 = tf.nn.conv2d(maxpool_conv1, filter_conv2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.depthwise_conv2d(maxpool_conv1, filter_conv2, strides=[1, 1, 1, 1], padding='SAME')

    # to avoid negative numbers in our matrices we set each value < 0 to 0 with the relu operation
    relu_conv2 = tf.nn.relu(conv2 + filter_bias_conv2)

    # to reduce the pixels in our image we maxpool => this is to ignore unimportant pixels
    maxpool_conv2 = tf.nn.max_pool(relu_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    ###############Fully Connected NN######################

    #create a weight and a bias for each Neuron
    nrOfInputNeurons = 9720  # 27040
    nrOfOutputPixels = 9*9*number_of_filters*number_of_filters_conv2*3 #9720
    W_fc1 = tf.Variable(tf.truncated_normal([nrOfOutputPixels, nrOfInputNeurons], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[nrOfInputNeurons]))

    #flattern out input images
    h_pool2_flat = tf.reshape(maxpool_conv2, [-1, nrOfOutputPixels])

    #do the matrix multiplication
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # read out layer
    W_fc2 = tf.Variable(tf.truncated_normal([nrOfInputNeurons, 4], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[4]))


    ################################END###########################################################

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
    saver = tf.train.Saver(max_to_keep=1)

    latest_checkpoint = tf.train.latest_checkpoint("SavedCNN_Color")

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
    print("################ TRAINING STARTS #######################")
    for i in range(training_step,20000):
      #batch , labels , errors = myhandler.get_batch(100)
      batch, labels, errors = myhandler.load_color_files_and_get_batch(100)
      #print("Batch Len: " + str(len(batch)))

      #print("Batchsize: {0}".format(str(len(batch))))
      print("Step: %d" % i)
      if i%100 == 0:
          batch_classify, labels_classify, errors_classify = myhandler.load_files_and_get_batch_color_for_classification(100)
          train_accuracy = accuracy.eval(feed_dict={
              x: batch_classify, y_: labels_classify, keep_prob: 1.0})
          print("step %d, training accuracy %g" % (i, train_accuracy))
          saver.save(sess, 'SavedCNN_Color/AmpelPhasenCNN_Color', global_step=i)
      train_step.run(feed_dict={x: batch, y_: labels, keep_prob: 0.5})
    print("---------FINISHED--------")

if __name__ == "__main__":
    main()