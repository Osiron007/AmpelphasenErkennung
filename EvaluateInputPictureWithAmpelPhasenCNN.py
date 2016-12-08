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
    #x = tf.placeholder(tf.float32, shape=[None, 50,50])
    x = tf.placeholder(tf.float32, shape=[None, 50, 50])

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

    sess.run(tf.initialize_all_variables())

    ###################################################
    # TensorFlow Saver ################################
    ###################################################

    # standard implementation
    saver = tf.train.Saver(max_to_keep=2)

    latest_checkpoint = tf.train.latest_checkpoint("SavedAmpelPhasenCNN")

    saver.restore(sess, latest_checkpoint)

    ###################################################
    # TensorFlow SummeryWriter#########################
    ###################################################

    #creates a summary of the whole graph and saves it into folder "SavedGraphDefinitions"
    summary_writer = tf.train.SummaryWriter('SavedGraphDefinitions', sess.graph)

    ###################################################
    # Read classification indicator####################
    ###################################################

    image_green = cv.imread("/home/dlm/AmpelPhasen_Bilder/green_mini.png", 1)
    image_yellow = cv.imread("/home/dlm/AmpelPhasen_Bilder/yellow_mini.png", 1)
    image_yellow_red = cv.imread("/home/dlm/AmpelPhasen_Bilder/yellow_red_mini.png", 1)
    image_red = cv.imread("/home/dlm/AmpelPhasen_Bilder/red_mini.png", 1)

    classifyImages = True

    while classifyImages == True:

        #generate rnd number for image selection
        imageNumber = rnd.randint(1, 1000)

        #create path
        path = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Bilder/NewData/Frame"
        pathToFile = path + str(imageNumber) + ".jpg"

        print("File: " + pathToFile)

        ###################################################
        # Read image and convert it########################
        ###################################################


        colorImage = cv.imread(pathToFile,cv.IMREAD_COLOR)

        #resizedColorImage = cv.resize(colorImage,(500,500), interpolation=cv.INTER_CUBIC)
        resizedColorImage = cv.resize(colorImage, (500, 500), interpolation=cv.INTER_LINEAR)

        greyscaleImage = cv.cvtColor(colorImage,cv.COLOR_BGR2GRAY)

        resizedImage = cv.resize(greyscaleImage,(50,50), interpolation=cv.INTER_CUBIC)

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
        del greyscaleImage
        del batch
        del classification



if __name__ == "__main__":
    main()

