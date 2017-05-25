import tensorflow as tf
import h5py
import numpy as np

def main(_):

    # settings
    samples = 2572
    trainingSize = 1800
    batchSize = 1800
    nClasses = 57
    channels = 86 # ECoG channels
    timeSteps = 258 # time steps per sample

    #########
    # SETUP #
    #########

    # load data
    data = h5py.File("Data/EC2_blocks_1_8_9_15_76_89_105_CV_HG_align_window_-0.5_to_0.79_file_nobaseline.h5")
    x_data = data['Xhigh gamma'][:]
    y_predata = data['y'][:]

    # shaping data and changing to one hot format
    x_data = x_data[..., np.newaxis] # add that fourth dimension 
    y_data = np.zeros((samples, nClasses))
    y_data[np.arange(samples), y_predata] = 1

    # split test and train
    # CHECKPOINT - do this smarter, and use TensorFlow's batching functions
    x_train = x_data[0:trainingSize]
    x_test = x_data[trainingSize:]
    y_train = y_data[0:trainingSize]
    y_test = y_data[trainingSize:]

    # weight and bias initialization
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # pooling (downsampling) and stride parameters
    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

    def max_pool(x, size):
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

    #########
    # MODEL #
    #########

    # input
    x = tf.placeholder(tf.float32, [None, channels, timeSteps, 1])
    y_truth = tf.placeholder(tf.float32, shape=[None, nClasses])

    # conv layer 1
    W_conv1 = weight_variable([5, 70, 1, 32]) # compute 32 features for each patch
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1, stride=3) + b_conv1)
    h_pool1 = max_pool(h_conv1, size=2)
    print h_conv1.shape
    print h_pool1.shape

    # conv layer 2
    W_conv2 = weight_variable([5, 5, 32, 64]) # compute 64 features for each 5x5 patch
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, stride=2) + b_conv2)
    h_pool2 = max_pool(h_conv2, size=3)
    print h_conv2.shape
    print h_pool2.shape

    # densely connected layer
    layerSize = int(h_pool2.shape[1] * h_pool2.shape[2] * h_pool2.shape[3])
    W_fc1 = weight_variable([layerSize, 200])
    b_fc1 = bias_variable([200])
    h_pool2_flat = tf.reshape(h_pool2, [-1, layerSize])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout layer
    keep_prob = tf.placeholder(tf.float32) # can be turned off by setting to 1
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    W_fc2 = weight_variable([200, nClasses])
    b_fc2 = bias_variable([nClasses])
    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    ############
    # TRAINING #
    ############

    # loss minimization - uses ADAM optimizer instead of gradient descent
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_truth, logits=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # accuracy
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_truth,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initialize variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # run
    print("Starting training")
    for i in range(2000):
        batchN = i%(trainingSize/batchSize)
        x_batch = x_train[batchN*batchSize: (batchN+1)*batchSize]
        y_batch = y_train[batchN*batchSize: (batchN+1)*batchSize]
        if i%1 == 0: # print accuracy
           train_accuracy = accuracy.eval(feed_dict={x:x_batch, y_truth:y_batch, keep_prob:1.0})
           test_accuracy = accuracy.eval(feed_dict={x:x_test, y_truth:y_test, keep_prob:1.0})
           print("step %d, training accuracy %g, test accuracy %g"%(i, train_accuracy, test_accuracy))
        train_step.run(feed_dict={x:x_batch, y_truth:y_batch, keep_prob:0.5})

    # final accuracy
    print("test accuracy %g"%accuracy.eval(feed_dict={x:x_test, y_truth:y_test, keep_prob:1.0}))

if __name__ == '__main__':
  tf.app.run(main=main)
