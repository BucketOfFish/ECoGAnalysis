import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

############
# SETTINGS #
############

samples_per_class = 5000
n_classes = 57
n_samples = samples_per_class * n_classes
n_channels = 10 # ECoG channels
n_timesteps = 129 # time steps per sample

batch_size = 100
dropout_rate = 0.3
learning_rate = 0.01
epochs = 10 # when to stop training

input_filename = "../../Data/ECoG/005.h5"
shuffle = False
save_directory = "Outputs/002/"
overwrite = True

#########
# MODEL #
#########

def model():

    # Model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(n_channels, 5),
		     strides=(1,2),
		     activation='relu',
		     input_shape=(n_channels, n_timesteps, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(64, kernel_size=(1, 5),
		     strides=(1,2),
		     activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(128, kernel_size=(1, 5),
		     strides=(1,2),
		     activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
		  optimizer=keras.optimizers.Adadelta(),
		  metrics=['accuracy'])

    model.summary()
    return model
