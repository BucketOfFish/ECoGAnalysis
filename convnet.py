import matplotlib
matplotlib.use('Agg') # remove reliance on X-frame
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import numpy as np
import os

def main(_):

    ############
    # SETTINGS #
    ############

    samples = 2572 # number of total events
    nClasses = 57
    channels = 86 # ECoG channels
    timeSteps = 258 # time steps per sample

    trainingSize = 1800
    batchSize = 163

    useBestChannels = False # whether to use only a subset of ECoG channels
    bestChannels = [34, 27, 37, 36, 25, 38, 42, 33, 24, 23] # channels to use - (ordered worst to best, but it doesn't matter)

    netType = "FC" # options are "FC" and "conv"
    FCLayerSize = 798

    optimizerName = "momentum" # options are 'Adam' and 'momentum'
    learningRate = pow(10, -2.48)
    momentum = pow(10, -0.83)

    inputDropoutRate = 0.44
    dropoutRate = 0.80

    delta = pow(10, -4) # when to stop training

    #########
    # SETUP #
    #########

    # load data
    data = h5py.File("Data/EC2_blocks_1_8_9_15_76_89_105_CV_HG_align_window_-0.5_to_0.79_file_nobaseline.h5")
    x_data = data['Xhigh gamma'][:]
    y_predata = data['y'][:]

    # use only best channels
    if useBestChannels:
        channels = len(bestChannels)
        x_data = x_data[:,bestChannels,:]

    # shaping data and changing to one hot format
    x_data = x_data[..., np.newaxis] # add that fourth dimension 
    y_data = np.zeros((samples, nClasses))
    y_data[np.arange(samples), y_predata] = 1
    del y_predata

    # split test and train
    # CHECKPOINT - do this smarter, and use TensorFlow's batching functions
    x_train = x_data[0:trainingSize]
    x_test = x_data[trainingSize:]
    y_train = y_data[0:trainingSize]
    y_test = y_data[trainingSize:]

    # weight and bias initialization
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # pooling (downsampling) and stride parameters
    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

    def max_pool(x, size):
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

    #########
    # MODEL #
    #########

    # input
    x = tf.placeholder(tf.float32, [None, channels, timeSteps, 1])
    y_truth = tf.placeholder(tf.float32, shape=[None, nClasses])

    # not implemented - make the dropout layer a placeholder, so I can set to 1
    ## input dropout layer
    # keep_prob = tf.placeholder(tf.float32) # can be turned off by setting to 1
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    if netType == "conv":

        # conv layer 1
        W_conv1 = weight_variable([2, 70, 1, 32]) # compute 32 features for each patch
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1, stride=1) + b_conv1)
        h_pool1 = max_pool(h_conv1, size=2)
        print h_conv1.shape
        print h_pool1.shape

        # conv layer 2
        W_conv2 = weight_variable([2, 5, 32, 64]) # compute 64 features for each 5x5 patch
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, stride=1) + b_conv2)
        h_pool2 = max_pool(h_conv2, size=2)
        print h_conv2.shape
        print h_pool2.shape

        # densely connected layer
        layerSize = int(h_pool2.shape[1] * h_pool2.shape[2] * h_pool2.shape[3])
        W_fc1 = weight_variable([layerSize, FCLayerSize])
        b_fc1 = bias_variable([FCLayerSize])
        h_pool2_flat = tf.reshape(h_pool2, [-1, layerSize])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout layer
        keep_prob = tf.placeholder(tf.float32) # can be turned off by setting to 1
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    elif netType == "FC":

        # densely connected layer
        inputLayerSize = int(x.shape[1] * x.shape[2] * x.shape[3])
        W_fc1 = weight_variable([inputLayerSize, FCLayerSize])
        b_fc1 = bias_variable([FCLayerSize])
        x_flat = tf.reshape(x, [-1, inputLayerSize])
        h_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

        # dropout layer
        keep_prob = tf.placeholder(tf.float32) # can be turned off by setting to 1
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    W_fc2 = weight_variable([FCLayerSize, nClasses])
    b_fc2 = bias_variable([nClasses])
    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    ############
    # TRAINING #
    ############

    # set up optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_truth, logits=y))
    optimizerParameters = {'learning_rate': learningRate}
    optimizer = tf.train.GradientDescentOptimizer(*optimizerParameters)
    if optimizerName == "Adam":
        optimizer = tf.train.AdamOptimizer(*optimizerParameters)
    elif optimizerName == "momentum": # gradient descent with momentum
        optimizerParameters['momentum'] = momentum
        optimizer = tf.train.MomentumOptimizer(learning_rate=learningRate, momentum=momentum) # fix this
    train_step = optimizer.minimize(cross_entropy)

    # accuracy
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_truth,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initialize session and variables
    # sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True)) # print out CPU/GPU device usage
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # run
    print("Starting training")
    train_accuracies = []
    test_accuracies = []
    batchCount = 0
    #while len(train_accuracies) < 10 or abs(train_accuracies[-1] - train_accuracies[-10]) > delta:
    for i in range(5000):
        batchCount += 1
        batchN = batchCount%(trainingSize/batchSize)
        x_batch = x_train[batchN*batchSize: (batchN+1)*batchSize]
        y_batch = y_train[batchN*batchSize: (batchN+1)*batchSize]
        if batchCount%1 == 0: # print accuracy
           train_accuracy = accuracy.eval(feed_dict={x:x_batch, y_truth:y_batch, keep_prob:1.0})
           test_accuracy = accuracy.eval(feed_dict={x:x_test, y_truth:y_test, keep_prob:1.0})
           print("step %d, training accuracy %g, test accuracy %g"%(batchCount, train_accuracy, test_accuracy))
           train_accuracies.append(train_accuracy)
           test_accuracies.append(test_accuracy)
        train_step.run(feed_dict={x:x_batch, y_truth:y_batch, keep_prob:1-dropoutRate})

    # final accuracy
    print("test accuracy %g"%accuracy.eval(feed_dict={x:x_test, y_truth:y_test, keep_prob:1.0}))

    # plot accuracies
    indices = np.arange(len(train_accuracies))
    plt.plot(indices, train_accuracies)
    plt.plot(indices, test_accuracies)
    plt.legend(['train', 'test'])
    plt.title("Training and Test Accuracy vs. Training Steps")
    plt.xlabel("Training step")
    plt.ylabel("Accuracy")
    directory = "Plots/Training/"
    if not os.path.exists(directory): os.makedirs(directory)
    plt.savefig(directory + "Training.pdf", bbox_inches="tight")

    # save accuracies
    dataFile = h5py.File(directory + "TrainingData.h5", "w")
    dataFile.create_dataset('trainingAccuracy', data=train_accuracies)
    dataFile.create_dataset('testAccuracy', data=test_accuracies)
    dataFile.close()

if __name__ == '__main__':
    tf.app.run(main=main)
