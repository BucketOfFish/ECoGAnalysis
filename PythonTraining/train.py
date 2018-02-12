import h5py as h5
import numpy as np
import os
import skopt

############
# SETTINGS #
############

from Settings.settings_001 import *

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

# # don't know why I had to do this
# x_train = np.array(x_train)
# y_train = np.array(y_train)

#######
# RUN #
#######

# save this code
if not os.path.exists(save_directory): os.makedirs(save_directory)
else: print "Output folder", save_directory, "already exists - exiting"; exit()

# optimize training
# res = skopt.gp_minimize(Keras_train, [(50, 200), (0, 0.7)])
Keras_train(0)
# skopt.dump(res, "OptFile", store_objective=False, n_calls=10) # only do 10 passes, then warm-call with the last result to keep minimizing
