import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
import numpy as np
import os

# load data
data = h5py.File("Data/EC2_blocks_1_8_9_15_76_89_105_CV_HG_align_window_-0.5_to_0.79_file_nobaseline.h5")
X_data = data['Xhigh gamma'][:]
y_data = data['y'][:] # phoneme label

# preprocessing
X_data = np.divide(X_data, np.amax(X_data)) # normalize ECoG signal

# sort by phonemes
sort_order = y_data.argsort() # sort by phonemes
X_data = X_data[sort_order]
y_data = y_data[sort_order]

# take only best channels
bestChannels = [34, 27, 37, 36, 25, 38, 42, 33, 24, 23] # ordered worst to best
X_data = X_data[:,bestChannels,:]

# plot individual events
def plotEvent(eventN):
    X = X_data[eventN]
    y = y_data[eventN]

    plt.imshow(X, cmap=cm.gist_rainbow, vmin=0, vmax=1)
    plt.title("Event " + str(eventN) + " - ECoG signal during production of phoneme " + str(y))
    plt.xlabel("Time step")
    plt.ylabel("ECOG channel")
    directory = "Plots/BestChannels/Phoneme" + str(y) + "/"
    if not os.path.exists(directory): os.makedirs(directory)
    plt.savefig(directory + "Event" + str(eventN) + ".pdf", bbox_inches="tight")

for eventN in range(len(y_data)):
    plotEvent(eventN)

# # plot averages for each phoneme
# for phoneme in range(max(y_data)):
    # indices = [y_data == phoneme]
    # X = np.mean(X_data[indices], axis=0)

    # plt.imshow(X, cmap=cm.gist_rainbow, vmin=0, vmax=1)
    # plt.title("Average ECoG signal during production of phoneme " + str(phoneme))
    # plt.xlabel("Time step")
    # plt.ylabel("ECoG channel")
    # directory = "Plots/BestChannels/Averages/"
    # if not os.path.exists(directory): os.makedirs(directory)
    # plt.savefig(directory + "Phoneme" + str(phoneme) + ".pdf", bbox_inches="tight")
