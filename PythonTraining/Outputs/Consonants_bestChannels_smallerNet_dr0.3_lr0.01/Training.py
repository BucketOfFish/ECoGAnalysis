import matplotlib
matplotlib.use('Agg') # remove reliance on X-frame
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
from shutil import copyfile
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import random
import skopt

############
# SETTINGS #
############

samples = 28500 # number of total events
nClasses = 57
# nClasses = 19
channels = 86 # ECoG channels
timeSteps = 258 # time steps per sample

trainingSize = 28500 # test and training have already been split in file
epochs = 15 # when to stop training

useBestChannels = True # whether to use only a subset of ECoG channels
bestChannels = [34, 27, 37, 36, 25, 38, 42, 33, 24, 23] # channels to use - (ordered worst to best, but it doesn't matter)

dataFile = "../../../Data/ECoG/ExpandedIsolatedGaussian/Expanded_ECoG_285Isolated_GaussianNoise.h5"
saveName = "Consonants_bestChannels_smallerNet_dr0.3_lr0.01"

#############
# PREP DATA #
#############

# load data
data = h5py.File(dataFile)
#data = h5py.File("../../Data/EC2_blocks_1_8_9_15_76_89_105_CV_HG_align_window_-0.5_to_0.79_file_nobaseline.h5")
x_train = data['Xhigh gamma'][:]
y_pretrain = data['y'][:]
y_pretrain = np.array([int(i/3) for i in y_pretrain])
x_test = data['Xhigh gamma isolated'][:]
y_pretest = data['y isolated'][:]
y_pretest = np.array([int(i/3) for i in y_pretest])

# use only best channels
if useBestChannels:
    channels = len(bestChannels)
    x_train = x_train[:,bestChannels,:]
    x_test = x_test[:,bestChannels,:]

# shaping data and changing to one hot format
x_train = x_train[..., np.newaxis] # add dimension for CNN
x_test = x_test[..., np.newaxis] # add dimension for CNN
y_train = np.zeros((samples, nClasses))
y_train[np.arange(samples), y_pretrain] = 1
y_test = np.zeros((y_pretest.shape[0], nClasses))
y_test[np.arange(y_pretest.shape[0]), y_pretest] = 1
del y_pretrain, y_pretest, data

# shuffle
z = zip(x_train, y_train)
random.shuffle(z)
x_train, y_train = zip(*z)

# don't know why I had to do this
x_train = np.array(x_train)
y_train = np.array(y_train)

##############
# TENSORFLOW #
##############

weightInit = pow(10, -2.7) # Gaussian standard deviation

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=weightInit))

def bias_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=weightInit))

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def max_pool(x, size):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

############
# TRAINING #
############

def Keras_train(args):

    # batchSize = args[0]
    # dropoutRate = args[1]
    batchSize = 100
    dropoutRate = 0.3
    inputDropoutRate = 0
    FCLayerSize = 128
    learningRate = 0.01

    # Model
    model = Sequential()
    # model.add(MaxPooling2D(pool_size=(2,1), strides=(1,1), input_shape=(channels, timeSteps, 1))
    # model.add(Conv2D(32, kernel_size=(1,2), strides=(1,1), activation='relu'))
    model.add(Conv2D(16, kernel_size=(1,2), strides=(1,2), activation='relu', input_shape=(channels, timeSteps, 1)))
    # model.add(MaxPooling2D(pool_size=(2,1), strides=(1,1)))
    model.add(Dropout(dropoutRate))
    model.add(Conv2D(32, kernel_size=(1,3), strides=(1,3), activation='relu'))
    model.add(Conv2D(2, kernel_size=(1,1), strides=(1,1), activation='relu'))
    model.add(Dropout(dropoutRate))
    model.add(Conv2D(8, kernel_size=(1,43), strides=(1,43), activation='relu'))
    model.add(Dropout(dropoutRate))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropoutRate))
    model.add(Dense(nClasses, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.summary()

    # Train
    model.fit(x_train, y_train,
              batch_size=batchSize,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save(directory + "Model.h5")

def tf_train(args):

    batchSize = args[0]
    dropoutRate = args[1]
    inputDropoutRate = 0
    FCLayerSize = 128
    learningRate = 0.01

    # Model
    x = tf.placeholder(tf.float32, [None, channels, timeSteps, 1])
    y_truth = tf.placeholder(tf.float32, shape=[None, nClasses])
    input_keep_prob = tf.placeholder(tf.float32) # can be turned off by setting to 1
    x_drop = tf.nn.dropout(x, input_keep_prob)
    # conv layer 1
    W_conv1 = weight_variable([1, 2, 1, 32]) # compute 32 features for each patch
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_drop, W_conv1, stride=1) + b_conv1)
    # dropout layer
    keep_prob = tf.placeholder(tf.float32) # can be turned off by setting to 1
    h_1 = tf.nn.dropout(h_conv1, keep_prob)
    # conv layer 2
    W_conv2 = weight_variable([1, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_1, W_conv2, stride=1) + b_conv2)
    # dropout layer
    h_2 = tf.nn.dropout(h_conv2, keep_prob)
    # conv layer 3
    W_conv3 = weight_variable([1, 43, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_2, W_conv3, stride=1) + b_conv3)
    # dropout layer
    h_3 = tf.nn.dropout(h_conv3, keep_prob)
    # densely connected layer
    layerSize = int(h_3.shape[1] * h_3.shape[2] * h_3.shape[3])
    W_fc1 = weight_variable([layerSize, FCLayerSize])
    b_fc1 = bias_variable([FCLayerSize])
    h_3_flat = tf.reshape(h_3, [-1, layerSize])
    h_fc1 = tf.nn.relu(tf.matmul(h_3_flat, W_fc1) + b_fc1)
    # dropout layer
    h_last = tf.nn.dropout(h_fc1, keep_prob)
    # readout layer
    W_readout = weight_variable([FCLayerSize, nClasses])
    b_readout = bias_variable([nClasses])
    y = tf.matmul(h_last, W_readout) + b_readout
    # optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_truth, logits=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    train_step = optimizer.minimize(cross_entropy)
    # accuracy
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_truth,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Training
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    train_accuracies = []
    test_accuracies = []
    batchCount = 0
    epochCount = 0
    nBatchesInEpoch = trainingSize/batchSize
    keepLooping = True
    while keepLooping:
        keepLooping = (epochCount <= epochs)
        batchN = batchCount % nBatchesInEpoch
        if batchN == 0:
            epochCount += 1
        x_batch = x_train[batchN*batchSize: (batchN+1)*batchSize] # CHECKPOINT - remainder not used? go through logic in this block
        y_batch = y_train[batchN*batchSize: (batchN+1)*batchSize]
        if batchCount % 100 == 0: # print accuracy
           train_accuracy = accuracy.eval(feed_dict={x:x_batch, y_truth:y_batch, input_keep_prob:1.0, keep_prob:1.0})
           test_accuracy = accuracy.eval(feed_dict={x:x_test, y_truth:y_test, input_keep_prob:1.0, keep_prob:1.0})
           #print("step %d, training accuracy %g, test accuracy %g"%(batchCount, train_accuracy, test_accuracy))
           train_accuracies.append(train_accuracy)
           test_accuracies.append(test_accuracy)
        batchCount += 1
        train_step.run(feed_dict={x:x_batch, y_truth:y_batch, input_keep_prob:1.0-inputDropoutRate, keep_prob:1.0-dropoutRate})

#######
# RUN #
#######

# save this code
directory = "Outputs/" + saveName + "/"
if not os.path.exists(directory): os.makedirs(directory)
else: print "Output folder", directory, "already exists - exiting"; exit()
copyfile(__file__, directory + "Training.py")

# optimize training
# res = skopt.gp_minimize(Keras_train, [(50, 200), (0, 0.7)])
Keras_train(0)
# skopt.dump(res, "OptFile", store_objective=False, n_calls=10) # only do 10 passes, then warm-call with the last result to keep minimizing
