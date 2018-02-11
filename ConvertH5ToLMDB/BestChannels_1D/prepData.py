import h5py as h5
import numpy as np

path = '/home/matt/Projects/Data/ECoG/ExpandedIsolatedGaussian/'
old = h5.File(path + "Expanded_ECoG_285Isolated_GaussianNoise.h5")
new = h5.File(path + "Expanded_ECoG_285Isolated_GaussianNoise_1D_Downsampled.h5")

goodChannels = [23, 24, 33, 42, 38, 25, 36, 37, 27, 34]

X_train = old['Xhigh gamma'].value[:,goodChannels,3:-3].reshape(28500, 1, -1) # delete first and last 3 time steps to leave 252 (easier to divide into) 
X_test = old['Xhigh gamma isolated'].value[:,goodChannels,3:-3].reshape(285, 1, -1)

# average every two values together (this is acceptable because one ECoG row has 258 timesteps)
X_train = np.mean(X_train.reshape(-1, 2), axis=1)
X_test = np.mean(X_test.reshape(-1, 2), axis=1)

X_train = X_train.reshape(28500, 1, -1)
X_test = X_test.reshape(285, 1, -1)

new.create_dataset('Xhigh gamma', data=X_train)
new.create_dataset('Xhigh gamma isolated', data=X_test)
new.create_dataset('y', data=old['y'])
new.create_dataset('y isolated', data=old['y isolated'])
