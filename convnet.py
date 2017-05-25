import tensorflow as tf
import h5py

def main(_):

    # load data
    data = h5py.File("Data/EC2_blocks_1_8_9_15_76_89_105_CV_HG_align_window_-0.5_to_0.79_file_nobaseline.h5")
    x_data = data['Xhigh gamma'][:]
    y_data = data['y'][:] # phoneme label

    # weight and bias initialization
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # pooling (downsampling) and stride parameters
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #########
    # MODEL #
    #########

    # input
    x = tf.placeholder(tf.float32, [None, 784])
    y_truth = tf.placeholder(tf.float32, shape=[None, 10])

    # conv layer 1
    W_conv1 = weight_variable([5, 5, 1, 32]) # compute 32 features for each 5x5 patch
    b_conv1 = bias_variable([32])
    x_2D = tf.reshape(x, [-1, 28, 28, 1]) # reshape to 28x28 image, with 1 color channel
    h_conv1 = tf.nn.relu(conv2d(x_2D, W_conv1) + b_conv1) # 24x24x32?
    h_pool1 = max_pool_2x2(h_conv1) # 14x14x32

    # conv layer 2
    W_conv2 = weight_variable([5, 5, 32, 64]) # compute 64 features for each 5x5 patch
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # 10x10x64?
    h_pool2 = max_pool_2x2(h_conv2) # 7x7x64

    # densely connected layer
    W_fc1 = weight_variable([7*7*64, 1024]) # 1024 neuron output
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout layer
    keep_prob = tf.placeholder(tf.float32) # can be turned off by settign to 1
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    W_fc2 = weight_variable([1024, 10]) # 10 outputs for 10 images
    b_fc2 = bias_variable([10])
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
    for i in range(2000):
        batch = data.train.next_batch(50)
        if i%100 == 0: # print accuracy
           train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_truth: batch[1], keep_prob: 1.0})
           print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_truth: batch[1], keep_prob: 0.5})

    # final accuracy
    print("test accuracy %g"%accuracy.eval(feed_dict={x: data.test.images, y_truth: data.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  tf.app.run(main=main)