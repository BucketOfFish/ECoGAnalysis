import h5py as h5
import numpy as np
import os, shutil
import skopt
import pandas as pd
import keras
import tensorflow as tf

############
# SETTINGS #
############

from Settings.settings_001 import *

#######
# GPU #
#######

config = tf.ConfigProto(device_count = {'GPU': 1, 'CPU': 56})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

#############
# PREP DATA #
#############

# load data
data = h5.File(input_filename)
x_train = data['Xhigh gamma'][:]
y_pretrain = data['y'][:]
x_test = data['Xhigh gamma isolated'][:]
y_pretest = data['y isolated'][:]

# shaping data and changing to one hot format
x_train = x_train[..., np.newaxis] # add dimension for CNN
x_test = x_test[..., np.newaxis] # add dimension for CNN
y_train = np.zeros((n_samples, n_classes))
y_train[np.arange(n_samples), y_pretrain] = 1
y_test = np.zeros((y_pretest.shape[0], n_classes))
y_test[np.arange(y_pretest.shape[0]), y_pretest] = 1
del y_pretrain, y_pretest, data

# shuffle
if (shuffle):
    z = zip(x_train, y_train)
    np.random.shuffle(z)
    x_train, y_train = zip(*z)
    # x_train = np.array(x_train)
    # y_train = np.array(y_train)

#########
# TRAIN #
#########

# check if save directory already exists
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
else:
    if (overwrite):
        shutil.rmtree(save_directory)
        os.makedirs(save_directory)
    else:
        print "Output folder", save_directory, "already exists - exiting"
        exit()

# get model
model = model()

# train and save results
history = model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))
model.save(save_directory + "model.h5")
pd.DataFrame(history.history).to_csv(save_directory + "history.csv")
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
