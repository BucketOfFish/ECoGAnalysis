from __future__ import print_function
import h5py as h5
import numpy as np
from sklearn.utils import shuffle
import sys, os
import matplotlib.pyplot as plt

###########
# OPTIONS #
###########

from Settings.settings_002 import *

#############
# GROUPINGS #
#############

def grouping(label):

    """Given a CV label in (0-56), returns the consonant (0-18),
    vowel (0-2), place (0-2), and manner (0-2) labels.
    Note that not all CVs have place and manner labels (will return None).
    """
    consonant = label // 3
    vowel = label % 3

    if consonant in [0, 2, 10]:
        place = 0
    elif consonant in [1, 11, 6]:
        place = 1
    elif consonant in [3, 17]:
        place = 2
    else:
        place = None

    if consonant in [0, 1, 3]:
        manner = 0
    elif consonant in [2, 11]:
        manner = 1
    elif consonant in [10, 6, 17]:
        manner = 2
    else:
        manner = None

    return consonant, vowel, place, manner

########################
# AUGMENTATION METHODS #
########################

def interpolation(original_samples):
    n_original_samples = len(original_samples)
    i = np.random.randint(n_original_samples)
    j = np.random.randint(n_original_samples)
    if j==i: j = (j+1) % n_original_samples
    mixing = np.random.uniform()
    sample = original_samples[i]*mixing + original_samples[j]*(1-mixing)
    return sample

def gaussian_noise(sample):
    noise = np.random.randn(sample.shape[0], sample.shape[1]) * gaussian_noise_sigma
    sample = sample + noise
    return sample

def time_shift(sample):
    time_steps = np.random.randint(max_steps_timeshift) + 1
    positive_direction = bool(np.random.randint(2))
    zeros = np.zeros((sample.shape[0], time_steps))
    if (positive_direction):
        sample = np.concatenate((sample[:,-time_steps:], sample[:,:-time_steps]), axis=1)
    else:
        sample = np.concatenate((sample[:,time_steps:], sample[:,:time_steps]), axis=1)
    return sample

def amplitude_scale(sample):
    amplitude = np.random.uniform(min_amplitude_scale, max_amplitude_scale)
    sample = sample * amplitude
    return sample

def generateMoreSamples(original_samples):
    n_samples = len(original_samples)
    new_samples = []
    while n_samples < total_samples_per_class:
        if (do_interpolation):
            # only use original examples to ensure we don't get closer and closer to the mean
            new_sample = interpolation(original_samples)
        else:
            new_sample = original_samples[np.random.randint(len(original_samples))]
        # plt.imshow(new_sample); plt.colorbar(); plt.show()
        if (do_gaussian_noise):
            new_sample = gaussian_noise(new_sample)
        # plt.imshow(new_sample); plt.colorbar(); plt.show()
        if (do_time_shift):
            new_sample = time_shift(new_sample)
        # plt.imshow(new_sample); plt.colorbar(); plt.show()
        # make sure the sample is positive
        new_sample = (offset_multiplicative * (new_sample + offset_scalar)).clip(min=0)
        # plt.imshow(new_sample); plt.colorbar(); plt.show()
        if (do_amplitude_scale):
            new_sample = amplitude_scale(new_sample)
        # plt.imshow(new_sample); plt.colorbar(); plt.show()
        new_samples.append(new_sample)
        n_samples += 1
    return list(original_samples) + new_samples

#####################################
# PERFORM AUGMENTATION AND GROUPING #
#####################################

# make sure save file doesn't already exist
if (os.path.exists(new_filename) and not overwrite):
    print("Save file already exists")
    sys.exit()

# read in original data
if (not os.path.exists(original_filename)):
    print("Input file does not exist")
    sys.exit()
print("Reading in original samples")
data = h5.File(original_filename)
x = data['Xhigh gamma'][:]
y = data['y'][:]
if (use_best_channels):
    x = x[:,best_channels,:]
if (trim):
    x = x[:,:,1:-1]
if (downsample_factor > 1):
    if (downsample_factor == 2):
        x = (x[:,:,::downsample_factor] + x[:,:,1::downsample_factor]) / 2
    else:
        x = x[:,:,::downsample_factor]
if (flatten_1D):
    x = x.reshape(x.shape[0], 1, -1)

x_augmented = []
y_augmented = []
x_isolated = []
y_isolated = []

# isolate samples for each class, then use other samples to generate new samples
print("Generating new samples")
for i in range(57):
    if i%5==0:
        print('.', end='')
        sys.stdout.flush()
    indices = np.where(y==i)[0]
    x_class = x[indices]
    x_class = shuffle(x_class)
    x_isolated += list(x_class[:n_isolated_samples])
    y_isolated += [i]*n_isolated_samples
    new_x = generateMoreSamples(x_class[n_isolated_samples:])
    x_augmented += new_x
    y_augmented += [i]*len(new_x)
print("\nFinished generating new samples")
x = np.array(x_augmented)
y = np.array(y_augmented)
x_isolated = np.array(x_isolated)
y_isolated = np.array(y_isolated)
print("Finished converting arrays")

# shuffle samples
# To prevent a memory error, shuffle front and back halves separately
print("Shuffling and saving samples")
a = x[:15000]
b = y[:15000]
a, b = shuffle(a, b)
x[:15000] = a
y[:15000] = b
a = x[15000:]
b = y[15000:]
a, b = shuffle(a, b)
x[15000:] = a
y[15000:] = b

# save samples
new_data = h5.File(new_filename, "w")
new_data.create_dataset('Xhigh gamma', data=x)
new_data.create_dataset('Xhigh gamma isolated', data=x_isolated)
new_data.create_dataset('y', data=y)
new_data.create_dataset('y isolated', data=y_isolated)

# perform grouping
groupings = [grouping(i) for i in y]
y_consonant = [i[0] for i in groupings]
y_vowel = [i[1] for i in groupings]
y_place = [i[2] for i in groupings]
y_manner = [i[3] for i in groupings]
groupings_isolated = [grouping(i) for i in y_isolated]
y_consonant_isolated = [i[0] for i in groupings_isolated]
y_vowel_isolated = [i[1] for i in groupings_isolated]
y_place_isolated = [i[2] for i in groupings_isolated]
y_manner_isolated = [i[3] for i in groupings_isolated]
new_data.create_dataset('y_consonant', data=y_consonant)
new_data.create_dataset('y_consonant isolated', data=y_consonant_isolated)
new_data.create_dataset('y_vowel', data=y_vowel)
new_data.create_dataset('y_vowel isolated', data=y_vowel_isolated)
# new_data.create_dataset('y_place', data=y_place)
# new_data.create_dataset('y_place isolated', data=y_place_isolated)
# new_data.create_dataset('y_manner', data=y_manner)
# new_data.create_dataset('y_manner isolated', data=y_manner_isolated)
new_data.close()
