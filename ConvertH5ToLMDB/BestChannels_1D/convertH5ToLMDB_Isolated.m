% Matlab code

% This is the H5 file version where test and training samples are already pre-split
path = '/home/matt/Projects/Data/ECoG/ExpandedIsolatedGaussian/';
h5Name = strcat(path, 'Expanded_ECoG_285Isolated_GaussianNoise_1D_Downsampled.h5');
LMDBTest = strcat(path, 'BestChannels_1D_Downsampled/ECoG_test');
LMDBTrain = strcat(path, 'BestChannels_1D_Downsampled/ECoG_train');

X_test = h5read(h5Name, '/Xhigh gamma isolated');
y_test = h5read(h5Name, '/y isolated');
X_train = h5read(h5Name, '/Xhigh gamma');
y_train = h5read(h5Name, '/y');
X_test = permute(X_test, [1, 2, 4, 3]); % add dimension
X_train = permute(X_train, [1, 2, 4, 3]); % add dimension
y_test = y_test'; % transpose
y_train = y_train'; % transpose

% Cast data types
X_test = single(X_test); % cast from uint8 to float
y_test = int32(y_test);
X_train = single(X_train);
y_train = int32(y_train);

% Randomly shuffle data order.
nSamples_test = size(X_test, 4);
perm_test = randperm(nSamples_test);
X_test = X_test(:,:,:,perm_test);
y_test = y_test(perm_test)+1; % Matlab starts counting at 1
nSamples_train = size(X_train, 4);
perm_train = randperm(nSamples_train);
X_train = X_train(:,:,:,perm_train);
y_train = y_train(perm_train)+1; % Matlab is stupid

if ~exist(LMDBTest, 'dir')
    mkdir(LMDBTest);
end
clear write_lmdb
write_lmdb(LMDBTest, X_test, y_test, 'single');

if ~exist(LMDBTrain, 'dir')
    mkdir(LMDBTrain);
end
clear write_lmdb
write_lmdb(LMDBTrain, X_train, y_train, 'single');
