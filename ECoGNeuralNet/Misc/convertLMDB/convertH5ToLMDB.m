% Matlab code

path = '/home/matt/Projects/Data/ECoG/ExpandedIsolatedGaussian/';
h5Name = path + 'Expanded_ECoG_285Isolated_GaussianNoise.h5';
LMDBTest = path + 'ECoG_test';
LMDBTrain = path + 'ECoG_train';
nTest = 2000;
nTrain = 572;

X = h5read(h5Name, '/Xhigh gamma');
y = h5read(h5Name, '/y');
X = permute(X, [1, 2, 4, 3]); % add dimension
y = y'; % transpose

% Randomly shuffle data order.
nSamples = size(X, 4);
perm = randperm(nSamples);
X = X(:,:,:,perm);
y = y(perm);

X = single(X); % cast from uint8 to float
y = int32(y);

% Keep only these ECoG channels
goodChannels = [24, 25, 34, 43, 39, 26, 37, 38, 28, 35]

if ~exist(LMDBTest, 'dir')
    mkdir(LMDBTest);
end
clear write_lmdb
write_lmdb(LMDBTest, X(:,goodChannels,:,[1:nTest]), y(:,[1:nTest]), 'single');

if ~exist(LMDBTrain, 'dir')
    mkdir(LMDBTrain);
end
clear write_lmdb
write_lmdb(LMDBTrain, X(:,:,:,[nTest+1:nTest+nTrain]), y(:,[nTest+1:nTest+nTrain]), 'single');
