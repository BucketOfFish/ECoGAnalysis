import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
import numpy as np

# load data
data = h5py.File("EC2_blocks_1_8_9_15_76_89_105_CV_HG_align_window_-0.5_to_0.79_file_nobaseline.h5")
X_data = data['Xhigh gamma'][:]
y_data = data['y'][:] # phoneme label

# preprocessing
X_data = np.divide(X_data, np.amax(X_data)) # normalize EEG signal

# sort by phonemes
sort_order = y_data.argsort() # sort by phonemes
X_data = X_data[sort_order]
y_data = y_data[sort_order]

def plotEvent(eventN):
    X = X_data[eventN]
    y = y_data[eventN]

    plt.imshow(X, cmap=cm.gist_rainbow, vmin=0, vmax=1)
    plt.title("Event " + str(eventN) + " - EEG signal during production of phoneme " + str(y))
    plt.xlabel("Time step")
    plt.ylabel("ECOG channel")
    plt.show()


for eventN in range(1000):
    plotEvent(eventN)
