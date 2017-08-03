import matplotlib
matplotlib.use('Agg') # remove reliance on X-frame
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
from shutil import copyfile
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

def main():

    ############
    # SETTINGS #
    ############

    samples = 2572 # number of total events
    nClasses = 57
    channels = 86 # ECoG channels
    timeSteps = 258 # time steps per sample

    trainingSize = 1800
    batchSize = 10
    epochs = 100 # when to stop training
    dropoutRate = 0.50

    saveName = "CNN_AllChannels_1x1Layers"

    useBestChannels = False # whether to use only a subset of ECoG channels
    bestChannels = [34, 27, 37, 36, 25, 38, 42, 33, 24, 23] # channels to use - (ordered worst to best, but it doesn't matter)

    #########
    # MODEL #
    #########

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(channels, 5),
		     strides=(1,2),
		     activation='relu',
		     input_shape=(channels, timeSteps, 1)))
    model.add(Dropout(dropoutRate))
    model.add(Conv2D(64, kernel_size=(1, 5),
		     strides=(1,2),
		     activation='relu'))
    model.add(Dropout(dropoutRate))
    model.add(Conv2D(16, kernel_size=(1, 1),
		     strides=(1,2),
		     activation='relu'))
    model.add(Dropout(dropoutRate))
    model.add(Conv2D(32, kernel_size=(1, 5),
		     strides=(1,2),
		     activation='relu'))
    model.add(Dropout(dropoutRate))
    model.add(Conv2D(64, kernel_size=(1, 5),
		     strides=(1,2),
		     activation='relu'))
    model.add(Dropout(dropoutRate))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropoutRate))
    model.add(Dense(nClasses, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
		  optimizer=keras.optimizers.Adadelta(),
		  metrics=['accuracy'])

    model.summary()

    #########
    # SETUP #
    #########

    # save this code
    directory = "Outputs/" + saveName + "/"
    if not os.path.exists(directory): os.makedirs(directory)
    else: print "Output folder", directory, "already exists - exiting"; exit()
    copyfile("Keras_nets.py", directory + "Keras_nets.py")

    # load data
    data = h5py.File("../../Data/EC2_blocks_1_8_9_15_76_89_105_CV_HG_align_window_-0.5_to_0.79_file_nobaseline.h5")
    x_data = data['Xhigh gamma'][:]
    y_predata = data['y'][:]

    # use only best channels
    if useBestChannels:
        channels = len(bestChannels)
        x_data = x_data[:,bestChannels,:]

    # shaping data and changing to one hot format
    x_data = x_data[..., np.newaxis] # add that fourth dimension 
    y_data = keras.utils.to_categorical(y_predata, nClasses)
    del y_predata

    # shuffle data
    p = np.random.permutation(len(x_data))
    x_data = x_data[p]
    y_data = y_data[p]

    # split test and train
    x_train = x_data[0:trainingSize]
    x_test = x_data[trainingSize:]
    y_train = y_data[0:trainingSize]
    y_test = y_data[trainingSize:]

    ############
    # TRAINING #
    ############

    # run
    model.fit(x_train, y_train,
	      batch_size=batchSize,
	      epochs=epochs,
	      verbose=1,
	      validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # save accuracies
    model.save(directory + "Model.h5")

if __name__ == '__main__':
    main()
