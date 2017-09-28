import matplotlib
matplotlib.use('Agg') # remove reliance on X-frame
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import numpy as np
import os
from shutil import copyfile
import random

def main(_):

    ############
    # SETTINGS #
    ############

    samples = 2572 # number of total events
    nClasses = 57
    channels = 86 # ECoG channels
    timeSteps = 258 # time steps per sample

    trainingSize = int(samples * 0.8)
    batchSize = 163

    useReducedData = True # whether to use only a subset of ECoG data - not compatible with conv net!
    reducedInputs = [39, 257, 388, 422, 434, 552, 611, 626, 834, 1066, 1074, 1087, 1110, 1361, 1369, 1398, 1415, 1428, 1477,
    1572, 1580, 1586, 1622, 1687, 1710, 1793, 1826, 1872, 1873, 1873, 1886, 1912, 1918, 1995, 2089, 2119, 2127, 2149, 2209,
    2209, 2220, 2240, 2383, 2603, 2608, 2677, 2788, 2805, 2967, 3029, 3122, 3169, 3280, 3345, 3399, 3416, 3418, 3418, 3554,
    3639, 3737, 3746, 3982, 3989, 4030, 4064, 4079, 4256, 4267, 4401, 4434, 4478, 4501, 4635, 4687, 4762, 4780, 4781, 4851,
    4856, 4857, 4861, 4958, 5032, 5226, 5281, 5352, 5459, 5540, 5549, 5709, 5738, 5753, 5767, 5795, 5807, 5875, 5939, 5950,
    6004, 6010, 6012, 6018, 6028, 6035, 6064, 6095, 6219, 6246, 6267, 6297, 6297, 6310, 6313, 6335, 6436, 6553, 6567, 6590,
    6711, 6728, 6785, 6842, 6846, 6862, 6990, 7072, 7090, 7140, 7302, 7309, 7315, 7342, 7343, 7356, 7655, 7714, 7832, 7909,
    8069, 8075, 8294, 8377, 8405, 8442, 8479, 8498, 8542, 8577, 8590, 8592, 8603, 8611, 8612, 8615, 8616, 8623, 8624, 8626,
    8636, 8650, 8652, 8654, 8659, 8668, 8735, 8737, 8752, 8802, 8819, 8871, 8884, 8892, 8895, 8897, 8918, 8919, 8926, 8951,
    8991, 9129, 9267, 9277, 9331, 9333, 9371, 9375, 9398, 9405, 9420, 9421, 9463, 9498, 9522, 9625, 9635, 9649, 9676, 9704,
    9759, 9860, 9882, 9901, 9946, 9994, 10018, 10173, 10195, 10245, 10362, 10378, 10379, 10524, 10558, 10660, 10669, 10672,
    10701, 10718, 10719, 10732, 10804, 10820, 10835, 10879, 10881, 10900, 10930, 10931, 10949, 10954, 10955, 10955, 10955,
    10965, 10967, 10967, 10967, 10971, 10972, 10974, 10975, 10984, 10985, 10995, 10997, 11008, 11010, 11017, 11019, 11027,
    11043, 11094, 11140, 11173, 11188, 11211, 11223, 11232, 11235, 11236, 11250, 11258, 11273, 11281, 11315, 11410, 11413, 
    11435, 11488, 11512, 11514, 11558, 11573, 11630, 11693, 11694, 11704, 11710, 11718, 11720, 11720, 11728, 11729, 11736, 
    11739, 11741, 11746, 11755, 11761, 11765, 11783, 11796, 11811, 11817, 11829, 11838, 11843, 11945, 11949, 11965, 12009, 
    12034, 12043, 12048, 12107, 12167, 12243, 12253, 12257, 12271, 12302, 12359, 12362, 12408, 12473, 12558, 12687, 12732, 
    12741, 12865, 12958, 12984, 12990, 13055, 13055, 13060, 13081, 13136, 13153, 13192, 13213, 13316, 13358, 13360, 13388, 
    13454, 13479, 13501, 13534, 13553, 13605, 13643, 13755, 13769, 13769, 13771, 13818, 13825, 13841, 13854, 13861, 13865, 
    13943, 13982, 13993, 14002, 14032, 14062, 14077, 14084, 14191, 14201, 14232, 14238, 14253, 14256, 14274, 14280, 14288,
    14296, 14298, 14333, 14395, 14448, 14515, 14550, 14555, 14558, 14571, 14586, 14586, 14592, 14600, 14612, 14639, 14650,
    14776, 14780, 14801, 14838, 14844, 14861, 14930, 14982, 15238, 15314, 15314, 15530, 15715, 15743, 15847, 16057, 16076,
    16089, 16106, 16111, 16117, 16119, 16287, 16341, 16547, 16552, 16560, 16583, 16587, 16588, 16589, 16634, 16635, 16641,
    16677, 16697, 16704, 16711, 16759, 16791, 16827, 16836, 16841, 16879, 16891, 16899, 16914, 16959, 16963, 16967, 17020,
    17071, 17107, 17183, 17192, 17223, 17225, 17303, 17326, 17363, 17372, 17375, 17389, 17406, 17443, 17445, 17517, 17566,
    17572, 17590, 17600, 17610, 17610, 17625, 17641, 17642, 17647, 17652, 17656, 17664, 17670, 17673, 17674, 17690, 17690,
    17695, 17704, 17710, 17727, 17730, 17732, 17737, 17737, 17738, 17745, 17752, 17787, 17822, 17848, 17881, 17976, 18031,
    18045, 18069, 18115, 18117, 18156, 18218, 18242, 18272, 18365, 18417, 18448, 18474, 18624, 18654, 18727, 18730, 18741,
    18865, 18874, 18944, 18963, 19000, 19032, 19048, 19108, 19109, 19129, 19156, 19169, 19173, 19260, 19304, 19327, 19330,
    19359, 19364, 19365, 19366, 19416, 19417, 19445, 19493, 19565, 19573, 19575, 19582, 19607, 19640, 19657, 19704, 19706,
    19740, 19741, 19766, 19777, 19782, 19797, 19857, 19913, 19915, 19920, 19922, 19930, 19967, 19978, 19978, 19983, 20027,
    20058, 20081, 20157, 20175, 20178, 20225, 20281, 20335, 20360, 20402, 20409, 20433, 20436, 20731, 20776, 20978, 21043,
    21054, 21123, 21136, 21180, 21229, 21326, 21396, 21406, 21453, 21460, 21475, 21515, 21535, 21579, 21598, 21744, 21780,
    21792, 21840, 21967, 22000, 22013, 22121, 22121, 22144, 22165] # pixels to keep

    netType = "FC2" # options are "FC", "FC2", and "conv"
    FCLayerSize = 100

    optimizerName = "Adadelta" # options are "Adam", "momentum", and "Adadelta"
    learningRate = pow(10, -3)
    momentum = pow(10, -2)

    inputDropoutRate = 0
    dropoutRate = 0.50

    weightInitType = "Gaussian" # options are "uniform" and "Gaussian"
    weightInit = pow(10, -2.7) # uniform value, or Gaussian standard deviation

    stopType = "epochs" # choices are "epochs" and "delta"
    epochs = 5000 # when to stop training
    delta = pow(10, -4)

    printoutPeriod = 100
    saveName = "inputDropout0_dropout0p5_Adadelta_FC2:100_lrE-3_batchSize163_useReducedDataTrue_weightGaussian"

    #########
    # SETUP #
    #########

    # save code
    directory = "Outputs/" + saveName + "/"
    if not os.path.exists(directory): os.makedirs(directory)
    copyfile("TensorFlow_nets.py", directory + "TensorFlow_nets.py")

    # load data
    data = h5py.File("../../Data/EC2_blocks_1_8_9_15_76_89_105_CV_HG_align_window_-0.5_to_0.79_file_nobaseline.h5")
    x_data = data['Xhigh gamma'][:]
    y_predata = data['y'][:]

    # shaping data and changing to one hot format
    if useReducedData:
        x_data = x_data.reshape(x_data.shape[0], -1) # make 1D
        x_data = x_data[:,reducedInputs] # use reduced input
        x_data = x_data[..., np.newaxis, np.newaxis] # add dimensions to make input work out
    else:
        x_data = x_data[..., np.newaxis] # add that fourth dimension for 2D conv 
    y_data = np.zeros((samples, nClasses))
    y_data[np.arange(samples), y_predata] = 1
    del y_predata

    # shuffle
    z = zip(x_data, y_data)
    random.shuffle(z)
    x_data, y_data = zip(*z)

    # split test and train
    # CHECKPOINT - do this smarter, and use TensorFlow's batching functions
    x_train = x_data[0:trainingSize]
    x_test = x_data[trainingSize:]
    y_train = y_data[0:trainingSize]
    y_test = y_data[trainingSize:]

    # weight and bias initialization
    def weight_variable(shape):
        if weightInitType == "Gaussian":
            initial = tf.truncated_normal(shape, stddev=weightInit)
        elif weightInitType == "uniform":
            initial = tf.ones(shape) * weightInit
        else:
            initial = tf.zeros(shape)
        return tf.Variable(initial)

    def bias_variable(shape):
        if weightInitType == "Gaussian":
            initial = tf.truncated_normal(shape, stddev=weightInit)
        elif weightInitType == "uniform":
            initial = tf.ones(shape) * weightInit
        else:
            initial = tf.zeros(shape)
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
    if useReducedData:
        x = tf.placeholder(tf.float32, [None, len(reducedInputs), 1, 1])
    y_truth = tf.placeholder(tf.float32, shape=[None, nClasses])

    # input dropout layer
    input_keep_prob = tf.placeholder(tf.float32) # can be turned off by setting to 1
    x_drop = tf.nn.dropout(x, input_keep_prob)

    if netType == "conv":

        # conv layer 1
        W_conv1 = weight_variable([2, 70, 1, 32]) # compute 32 features for each patch
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_drop, W_conv1, stride=1) + b_conv1)
        h_pool1 = max_pool(h_conv1, size=2)
        print h_conv1.shape
        print h_pool1.shape

        # conv layer 2
        W_conv2 = weight_variable([2, 5, 32, 64]) # compute 64 features for each 5x5 patch
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, stride=1) + b_conv2)
        h_pool2 = max_pool(h_conv2, size=2)
        print h_conv2.shape
        print h_pool2.shape

        # densely connected layer
        layerSize = int(h_pool2.shape[1] * h_pool2.shape[2] * h_pool2.shape[3])
        W_fc1 = weight_variable([layerSize, FCLayerSize])
        b_fc1 = bias_variable([FCLayerSize])
        h_pool2_flat = tf.reshape(h_pool2, [-1, layerSize])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout layer
        keep_prob = tf.placeholder(tf.float32) # can be turned off by setting to 1
        h_last = tf.nn.dropout(h_fc1, keep_prob)

    elif netType == "FC": # fully connected - 1 layer
 
        # densely connected layer
        inputLayerSize = int(x_drop.shape[1] * x_drop.shape[2] * x_drop.shape[3])
        W_fc1 = weight_variable([inputLayerSize, FCLayerSize])
        b_fc1 = bias_variable([FCLayerSize])
        x_flat = tf.reshape(x_drop, [-1, inputLayerSize])
        h_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

        # dropout layer
        keep_prob = tf.placeholder(tf.float32) # can be turned off by setting to 1
        h_last = tf.nn.dropout(h_fc1, keep_prob)

    elif netType == "FC2": # fully connected - 2 layers
 
        # densely connected layer
        inputLayerSize = int(x_drop.shape[1] * x_drop.shape[2] * x_drop.shape[3])
        W_fc1 = weight_variable([inputLayerSize, FCLayerSize])
        b_fc1 = bias_variable([FCLayerSize])
        x_flat = tf.reshape(x_drop, [-1, inputLayerSize])
        h_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

        # dropout layer
        keep_prob = tf.placeholder(tf.float32) # can be turned off by setting to 1
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
        # densely connected layer
        W_fc2 = weight_variable([FCLayerSize, FCLayerSize])
        b_fc2 = bias_variable([FCLayerSize])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        # dropout layer 2
        h_last = tf.nn.dropout(h_fc2, keep_prob)

    # readout layer
    W_readout = weight_variable([FCLayerSize, nClasses])
    b_readout = bias_variable([nClasses])
    y = tf.matmul(h_last, W_readout) + b_readout

    ############
    # TRAINING #
    ############

    # set up optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_truth, logits=y))
    if optimizerName == "Adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    elif optimizerName == "momentum": # gradient descent with momentum
        optimizer = tf.train.MomentumOptimizer(learning_rate=learningRate, momentum=momentum)
    elif optimizerName == "Adadelta":
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learningRate)
    else: # default gradient descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    train_step = optimizer.minimize(cross_entropy)

    # accuracy
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_truth,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initialize session and variables
    # sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True)) # print out CPU/GPU device usage
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # run
    print("Starting training")
    train_accuracies = []
    test_accuracies = []
    batchCount = 0
    epochCount = 0
    nBatchesInEpoch = trainingSize/batchSize

    keepLooping = True
    while keepLooping:

        if stopType == "epochs":
            keepLooping = (epochCount <= epochs)
        elif stopType == "delta":
            keepLooping = (len(train_accuracies) < 10 or abs(train_accuracies[-1] - train_accuracies[-10]) > delta) # CHECKPOINT - fix this

        batchN = batchCount % nBatchesInEpoch
        if batchN == 0:
            epochCount += 1
        x_batch = x_train[batchN*batchSize: (batchN+1)*batchSize] # CHECKPOINT - remainder not used? go through logic in this block
        y_batch = y_train[batchN*batchSize: (batchN+1)*batchSize]

        if batchCount % printoutPeriod == 0: # print accuracy
           train_accuracy = accuracy.eval(feed_dict={x:x_batch, y_truth:y_batch, input_keep_prob:1.0, keep_prob:1.0})
           test_accuracy = accuracy.eval(feed_dict={x:x_test, y_truth:y_test, input_keep_prob:1.0, keep_prob:1.0})
           print("step %d, training accuracy %g, test accuracy %g"%(batchCount, train_accuracy, test_accuracy))
           train_accuracies.append(train_accuracy)
           test_accuracies.append(test_accuracy)

        batchCount += 1
        train_step.run(feed_dict={x:x_batch, y_truth:y_batch, input_keep_prob:1-inputDropoutRate, keep_prob:1-dropoutRate})

    # plot accuracies
    indices = np.arange(len(train_accuracies))
    plt.plot(indices, train_accuracies)
    plt.plot(indices, test_accuracies)
    plt.legend(['train', 'test'])
    plt.title("Training and Test Accuracy vs. Training Steps")
    plt.xlabel("Training step")
    plt.ylabel("Accuracy")
    plt.savefig(directory + "Accuracy.pdf", bbox_inches="tight")

    # save accuracies
    dataFile = h5py.File(directory + "AccuracyData.h5", "w")
    dataFile.create_dataset('trainingAccuracy', data=train_accuracies)
    dataFile.create_dataset('testAccuracy', data=test_accuracies)
    dataFile.close()

if __name__ == '__main__':
    tf.app.run(main=main)
