# based on script from https://groups.google.com/forum/#!topic/caffe-users/2xpmLJYmt5k

from numpy import *
import lmdb
import h5py
import caffe

tgt_db_test = "/home/tn_user/MattWorkspace/ECoGAnalysis/Data/ECoG_test"
tgt_db_train = "/home/tn_user/MattWorkspace/ECoGAnalysis/Data/ECoG_train"
src_db = "/home/tn_user/MattWorkspace/ECoGAnalysis/Data/EC2_blocks_1_8_9_15_76_89_105_CV_HG_align_window_-0.5_to_0.79_file_nobaseline.h5"

train = lmdb.open(tgt_db_train, map_size=10000000000)
test = lmdb.open(tgt_db_test, map_size=10000000000)
nTrain = 2000
nTest = 572

with h5py.File(src_db,'r') as f:
    # extract data from hdf file
    ar_data = array(f['Xhigh gamma'],dtype=float32)
    ar_label = array(f['y'],dtype=int)
    n, h, w = ar_data.shape # n data points, height, width
    c = 1 # channels
    ar_label = ar_label.flatten()
    assert len(ar_label) == n # number of labels has to match the number of input images!
    # write data to lmdb
    with train.begin(write=True) as txn:
        for i in range(nTrain):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = c
            datum.height = h
            datum.width = w
            data = expand_dims(ar_data[i,:,:], axis=0)
            datum.data = data.tobytes()
            datum.label = ar_label[i]
            str_id = '{:08}'.format(i) # create an 8 digit string id based on the index
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
    with test.begin(write=True) as txn:
        for i in range(nTest):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = c
            datum.height = h
            datum.width = w
            data = expand_dims(ar_data[i+nTrain,:,:], axis=0)
            datum.data = data.tobytes()
            datum.label = ar_label[i+nTrain]
            str_id = '{:08}'.format(i) # create an 8 digit string id based on the index
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
