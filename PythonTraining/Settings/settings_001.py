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
save_directory = "Outputs/001/"

############
# TRAINING #
############

def Keras_train(args):

    # Model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(1,6), strides=(1,6), activation='relu', input_shape=(1, n_channels*n_timesteps, 1)))
    # model.add(Conv2D(16, kernel_size=(1,2), strides=(1,2), activation='relu', input_shape=(1, n_channels*n_timesteps, 1)))
    # model.add(Conv2D(32, kernel_size=(1,3), strides=(1,3), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(2, kernel_size=(1,1), strides=(1,1), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(2, kernel_size=(1,3), strides=(1,3), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(64, kernel_size=(1,7), strides=(1,7), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(12, kernel_size=(1,1), strides=(1,1), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(64, kernel_size=(1,10), strides=(1,10), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.summary()

    # Train
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save(directory + "Model.h5")
