import h5py as h5
import numpy as np
import os, shutil
import pandas as pd
import keras
import tensorflow as tf

############
# SETTINGS #
############

from Settings.settings_003_optimize import *

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
y_pretrain = data[y_samples][:]
x_test = data['Xhigh gamma isolated'][:]
y_pretest = data[y_samples+' isolated'][:]
n_samples = x_train.shape[0]

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

def train(layers):

    # get model
    my_model = model(layers)

    # train and save results
    history = my_model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test))
    my_model.save(save_directory + "model.h5")
    pd.DataFrame(history.history).to_csv(save_directory + "history.csv")
    score = my_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score

############
# OPTIMIZE #
############

layers_list = [
    [128, 1],
    [8, 16, 1],
    [2, 4, 16],
    [2, 4, 4, 4], 
    [4, 2, 4, 1, 4], 
    [2, 2, 2, 2, 2, 2, 2]]
scores = []

for layers in layers_list:
    scores.append(train(layers))

print scores
losses = [i[0] for i in scores]
accuracies = [i[1] for i in scores]
print "Highest test accuracy:", max(accuracies)
print "Best model:", layers_list[accuracies.index(max(accuracies))]

plot_convergence(result)
