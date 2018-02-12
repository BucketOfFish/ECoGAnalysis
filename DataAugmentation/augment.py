import h5py as h5
import numpy as np
from sklearn.utils import shuffle
import sys, os

###########
# OPTIONS #
###########

original_filename = "../../../Data/EC2_blocks_1_8_9_15_76_89_105_CV_HG_align_window_-0.5_to_0.79_file_nobaseline.h5"
new_filename = "../../../Data/Expanded_ECoG_285Isolated_GaussianNoise.h5"
overwrite = False # whether it's ok to overwrite an existing output file
n_isolated_samples = 5 # samples to keep separate and not use in augmentation (will later be used as test samples)
total_samples_per_class = 500 # total number of samples we want to end up with for each CV pair
interpolation = True
gaussian_noise = True
gaussian_noise_sigma = 0.5
time_shifting = True
max_steps_timeshift = 10
amplitude_scaling = True
min_amplitude_scale = 0.5
max_amplitude_scale = 2

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
    noise = np.random.randn(sample.size[0], sample.size[1]) * gaussian_noise_sigma
    sample = sample + noise
    sample = sample.clip(min=0)
    return sample

def time_shift(sample):
    time_steps = np.random.randint(max_steps_timeshift) + 1
    positive_direction = bool(np.random.randint(2))
    zeros = [0] * time_steps
    if (positive_direction):
        sample = zeros + sample[:-time_steps]
    else:
        sample = sample[time_steps:] + zeros
    return sample

def amplitude_scale(sample):
    amplitude = np.random.uniform(min_amplitude_scale, max_amplitude_scale)
    sample = sample * amplitude
    return sample

def generateMoreSamples(original_samples):
    n_samples = len(original_samples)
    new_samples = []
    while n_samples < total_samples_per_class:
        if (interpolation):
            # only use original examples to ensure we don't get closer and closer to the mean
            new_sample = interpolation(original_samples)
        else:
            new_sample = original_samples[np.random.randint(len(original_samples))]
        if (gaussian_noise):
            new_sample = gaussian_noise(new_sample)
        if (time_shift):
            new_sample = time_shift(new_sample)
        if (amplitude_scale):
            new_sample = amplitude_scale(new_sample)
        new_samples.append(new_sample)
        n_samples += 1
    return list(original_samples) + new_samples

########################
# PERFORM AUGMENTATION #
########################

# make sure save file doesn't already exist
if (os.path.exists(new_filename) and not overwrite):
    print "File already exists"
    sys.exit()

# read in original data
print "Reading in original samples"
data = h5.File(original_filename)
x = data['Xhigh gamma'][:]
y = data['y'][:]

x_augmented = []
y_augmented = []
x_isolated = []
y_isolated = []

# isolate samples for each class, then use other samples to generate new samples
for i in range(57):
    if i%10==0: print i
    indices = np.where(y==i)[0]
    x_isolated += list(x[indices[:n_isolated_samples]])
    y_isolated += [i]*n_isolated_samples
    new_x = generateMoreSamples(x[indices[n_isolated_samples:]])
    x_augmented += new_x
    y_augmented += [i]*len(new_x)
print "Finished generating new samples."
x = np.array(x_augmented)
print "Finished converting x array."
y = np.array(y_augmented)
print "Finished converting y array."
x_isolated = np.array(x_isolated)
y_isolated = np.array(y_isolated)
print "Finished converting isolated arrays."

# shuffle samples
print "Shuffling and saving samples"
x, y = shuffle(x, y, random_state=0)
# # To prevent a memory error, shuffle front and back halves separately
# a = x[:15000]
# b = y[:15000]
# a, b = shuffle(a, b, random_state=0)
# x[:15000] = a
# y[:15000] = b
# a = x[15000:]
# b = y[15000:]
# a, b = shuffle(a, b, random_state=0)
# x[15000:] = a
# y[15000:] = b

# save samples
new_data = h5.File(new_filename, "w")
new_data.create_dataset('Xhigh gamma', data=x)
new_data.create_dataset('y', data=y)
new_data.create_dataset('Xhigh gamma isolated', data=x_isolated)
new_data.create_dataset('y isolated', data=y_isolated)
new_data.close()
