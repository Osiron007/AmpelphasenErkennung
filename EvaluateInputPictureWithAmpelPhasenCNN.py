import sys

import tensorflow as tf

import cv2 as cv

import time

import random as rnd

import numpy as np

import Input_Data_Handler as IDH


def main():
    ###################################################
    # TensorFlow Graph definition#######################
    ###################################################

    # A TensorFlow Session for use in interactive contexts, such as a shell.
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/client.html#InteractiveSession
    sess = tf.InteractiveSession()

    # Placeholder
    # TensorFlow provides a placeholder operation that must be fed with data on execution.
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/io_ops.html#placeholders

    # tf.placeholder(dtype, shape=None, name=None)
    # dtype: The type of elements in the tensor to be fed.
    # shape: The shape of the tensor to be fed (optional). If the shape is not specified, you can feed a tensor of any shape.
    # name: A name for the operation (optional).

    # Placeholder for input batch data shape=[batch_size, NrOfPixel height, NrOfPixel weight, color channels]
    x = tf.placeholder(tf.float32, shape=[None, 50, 50, 3])

    # Placeholder for input labels shape=[batch_size, NrOfClasses]
    # Each label is a one hot vector: (1,0,0,0) ... (0,0,0,1)
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
    # to learn our features we need a variable for each pixel in each filter which will optimized during training
    filter_conv1 = tf.Variable(
        tf.truncated_normal([filter_height_conv1, filter_width_conv1, number_of_channels, number_of_filters],
                            stddev=0.1))

    # for each filter we need a bias variable which can be optimizied during training
    filter_bias_conv1 = tf.Variable(tf.constant(0.1, shape=[number_of_filters * number_of_channels]))

    ##### Convolution with tf.nn.conv2d ######################################################################
    # with this convolution layer we want to convolute out input image with a number of filters
    # the output will be multiple pictures, one for each filter

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
    # tf.nn.depthwise_conv2d(input, filter, strides, padding, name=None)
    # Args:
    # input: 4-D with shape [batch, in_height, in_width, in_channels].
    # filter: 4-D with shape [filter_height, filter_width, in_channels, channel_multiplier].
    # strides: 1-D of size 4. The stride of the sliding window for each dimension of input.
    # padding: A string, either 'VALID' or 'SAME'. The padding algorithm. See the comment here
    # name: A name for this operation (optional).

    # Returns:
    # A 4D Tensor of shape[batch, out_height, out_width, in_channels * channel_multiplier].

    conv1 = tf.nn.depthwise_conv2d(x_image, filter_conv1, strides=[1, 1, 1, 1], padding='SAME')
    ###########################################################################################################

    # to avoid negative numbers in our matrices we set each value < 0 to 0 with the relu operation
    relu_conv1 = tf.nn.relu(conv1 + filter_bias_conv1)

    # to reduce the pixels in our image we maxpool

    maxpool_conv1 = tf.nn.max_pool(relu_conv1, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

    ###############Second Convolution#################
    filter_height_conv2 = 3
    filter_width_conv2 = 3
    number_of_filters_conv2 = 5
    number_of_channels_conv2 = 24  # 3*8
    # to learn our features we need a variable for each pixel in each filter which will optimized during training
    filter_conv2 = tf.Variable(
        tf.truncated_normal(
            [filter_height_conv2, filter_width_conv2, number_of_channels_conv2, number_of_filters_conv2],
            stddev=0.1))

    # for each filter we need a bias variable which can be optimizied during training
    filter_bias_conv2 = tf.Variable(tf.constant(0.1, shape=[number_of_filters_conv2 * number_of_channels_conv2]))

    # with this convolution layer we want to convolute out input image with a number of filters
    # the output will be multiple pictures, one for each filter
    # conv2 = tf.nn.conv2d(maxpool_conv1, filter_conv2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.depthwise_conv2d(maxpool_conv1, filter_conv2, strides=[1, 1, 1, 1], padding='SAME')

    # to avoid negative numbers in our matrices we set each value < 0 to 0 with the relu operation
    relu_conv2 = tf.nn.relu(conv2 + filter_bias_conv2)

    # to reduce the pixels in our image we maxpool => this is to ignore unimportant pixels
    maxpool_conv2 = tf.nn.max_pool(relu_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    ###############Fully Connected NN######################

    # create a weight and a bias for each Neuron
    nrOfInputNeurons = 9720  # 27040
    nrOfOutputPixels = 9 * 9 * number_of_filters * number_of_filters_conv2 * 3  # 9720
    W_fc1 = tf.Variable(tf.truncated_normal([nrOfOutputPixels, nrOfInputNeurons], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[nrOfInputNeurons]))

    # flattern out input images
    h_pool2_flat = tf.reshape(maxpool_conv2, [-1, nrOfOutputPixels])

    # do the matrix multiplication
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # read out layer
    W_fc2 = tf.Variable(tf.truncated_normal([nrOfInputNeurons, 4], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[4]))

    ################################END###########################################################

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    sess.run(tf.initialize_all_variables())

    ###################################################
    # TensorFlow Saver ################################
    ###################################################

    # standard implementation
    saver = tf.train.Saver(max_to_keep=2)

    latest_checkpoint = tf.train.latest_checkpoint("SavedCNN_Color")

    saver.restore(sess, latest_checkpoint)

    ###################################################
    # TensorFlow SummeryWriter#########################
    ###################################################

    #creates a summary of the whole graph and saves it into folder "SavedGraphDefinitions"
    summary_writer = tf.train.SummaryWriter('SavedGraphDefinitions', sess.graph)

    ###################################################
    # Read classification indicator####################
    ###################################################

    image_green = cv.imread("/home/dlm/AmpelPhasen_Bilder/minipics/green_mini.png", 1)
    image_yellow = cv.imread("/home/dlm/AmpelPhasen_Bilder/minipics/yellow_mini.png", 1)
    image_yellow_red = cv.imread("/home/dlm/AmpelPhasen_Bilder/minipics/yellow_red_mini.png", 1)
    image_red = cv.imread("/home/dlm/AmpelPhasen_Bilder/minipics/red_mini.png", 1)

    classifyImages = True

    while classifyImages == True:

        #generate rnd number for image selection
        imageNumber = rnd.randint(1, 1000)

        #create path
        #path = "/home/dlm/AmpelPhasen_Bilder/Testbilder/Frame"

        path = "/home/dlm/AmpelPhasen_Bilder/Testbilder/grueneAmpel.jpg"
        #path = "/home/dlm/AmpelPhasen_Bilder/Testbilder/gelbeAmpel.jpg"
        #path = "/home/dlm/AmpelPhasen_Bilder/Testbilder/roteAmpel.jpg"
        #pathToFile = path + str(imageNumber) + ".jpg"

        #print("File: " + pathToFile)

        ###################################################
        # Read image and convert it########################
        ###################################################


        colorImage = cv.imread(path,cv.IMREAD_COLOR)

        #resizedColorImage = cv.resize(colorImage,(500,500), interpolation=cv.INTER_CUBIC)
        resizedColorImage = cv.resize(colorImage, (500, 500), interpolation=cv.INTER_LINEAR)

        #greyscaleImage = cv.cvtColor(colorImage,cv.COLOR_BGR2GRAY)

        resizedImage = cv.resize(resizedColorImage,(50,50), interpolation=cv.INTER_CUBIC)

        batch = []

        batch.append(resizedImage)

        #batch = resizedImage

        batch[0] = np.asarray(batch[0])

        #Evaluation
        print("Start Classification")
        #classification is from type numpy.ndarray
        classification = y_conv.eval(feed_dict={x: batch , keep_prob: 1.0})
        value_green = classification[0][0]
        value_yellow = classification[0][1]
        value_yellow_red = classification[0][3]
        value_red = classification[0][2]
        print("classification "+str(classification))
        print("Green Value " + str(value_green))
        print("Yellow Value " + str(value_yellow))
        print("YellowRed Value " + str(value_yellow_red))
        print("Red Value " + str(value_red))


        if value_green > value_yellow and value_green > value_yellow_red and value_green > value_red:
            print("The Trafficsign signals Green!!!")
            classificationImage = cv.resize(image_green,(80,80), interpolation=cv.INTER_CUBIC)

        if value_yellow > value_green and value_yellow > value_yellow_red and value_yellow > value_red:
            print("The Trafficsign signals Yellow!!!")
            classificationImage = cv.resize(image_yellow, (80, 80), interpolation=cv.INTER_CUBIC)

        if value_yellow_red > value_yellow and value_yellow_red > value_green and value_yellow_red > value_red:
            print("The Trafficsign signals Yellow Red!!!")
            classificationImage = cv.resize(image_yellow_red, (80, 80), interpolation=cv.INTER_CUBIC)

        if value_red > value_yellow and value_red > value_yellow_red and value_red > value_green:
            print("The Trafficsign signals Red!!!")
            classificationImage = cv.resize(image_red, (80, 80), interpolation=cv.INTER_CUBIC)

        #implement pictogram into real image
        for cnt_x in range(420,500,1):
            for cnt_y in range(420,500,1):
                for colour in range(0,3,1):
                    resizedColorImage[cnt_y][cnt_x][colour] = classificationImage[cnt_y-421][cnt_x-421][colour]


        #show picture with classification
        cv.imshow("Real Image", resizedColorImage)


        cv.waitKey(5000)

        #release images
        cv.destroyAllWindows()
        del colorImage
        #del resizedColorImage
        del classificationImage
        del resizedColorImage
        #del greyscaleImage
        del batch
        del classification



if __name__ == "__main__":
    main()

