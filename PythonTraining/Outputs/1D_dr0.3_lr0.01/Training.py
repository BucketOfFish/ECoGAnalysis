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
channels = 10 # ECoG channels
timeSteps = 126 # time steps per sample

trainingSize = 28500 # test and training have already been split in file
epochs = 10 # when to stop training

dataFile = "../../../Data/ECoG/ExpandedIsolatedGaussian/Expanded_ECoG_285Isolated_GaussianNoise_1D_Downsampled.h5"
saveName = "1D_dr0.3_lr0.01"

#############
# PREP DATA #
#############

# load data
data = h5py.File(dataFile)
x_train = data['Xhigh gamma'][:]
y_pretrain = data['y'][:]
x_test = data['Xhigh gamma isolated'][:]
y_pretest = data['y isolated'][:]

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
    model.add(Conv2D(32, kernel_size=(1,6), strides=(1,6), activation='relu', input_shape=(1, channels*timeSteps, 1)))
    # model.add(Conv2D(16, kernel_size=(1,2), strides=(1,2), activation='relu', input_shape=(1, channels*timeSteps, 1)))
    # model.add(Conv2D(32, kernel_size=(1,3), strides=(1,3), activation='relu'))
    model.add(Dropout(dropoutRate))
    model.add(Conv2D(2, kernel_size=(1,1), strides=(1,1), activation='relu'))
    model.add(Dropout(dropoutRate))
    model.add(Conv2D(2, kernel_size=(1,3), strides=(1,3), activation='relu'))
    model.add(Dropout(dropoutRate))
    model.add(Conv2D(64, kernel_size=(1,7), strides=(1,7), activation='relu'))
    model.add(Dropout(dropoutRate))
    model.add(Conv2D(12, kernel_size=(1,1), strides=(1,1), activation='relu'))
    model.add(Dropout(dropoutRate))
    model.add(Conv2D(64, kernel_size=(1,10), strides=(1,10), activation='relu'))
    model.add(Dropout(dropoutRate))
    model.add(Flatten())
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
