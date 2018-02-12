% Matlab code

input_filename = '/home/matt/Projects/Data/ECoG/007.h5';
samples_per_class = 5000;
n_classes = 57;
n_samples = samples_per_class * n_classes;
n_channels = 10;
n_timesteps = 128;

% Test and train are already split
X_test = h5read(input_filename, '/Xhigh gamma isolated');
y_test = h5read(input_filename, '/y isolated');
X_train = h5read(input_filename, '/Xhigh gamma');
y_train = h5read(input_filename, '/y');
X_test = permute(X_test, [1, 2, 4, 3]); % add dimension
X_train = permute(X_train, [1, 2, 4, 3]); % add dimension
y_test = y_test'; % transpose
y_train = y_train'; % transpose

% Cast data types
X_test = single(X_test); % cast from uint8 to float
y_test = int32(y_test);
X_train = single(X_train);
y_train = int32(y_train);

% Make 1D
X_test = reshape(X_test, 1, n_timesteps*n_channels, 1, size(X_test, 4));
X_train = reshape(X_train, 1, n_timesteps*n_channels, 1, size(X_train, 4));

% Randomly shuffle data order
nSamples_test = size(X_test, 4);
perm_test = randperm(nSamples_test);
X_test = X_test(:,:,:,perm_test);
y_test = y_test(perm_test)+1; % Matlab starts counting at 1
nSamples_train = size(X_train, 4);
perm_train = randperm(nSamples_train);
X_train = X_train(:,:,:,perm_train);
y_train = y_train(perm_train)+1; % Matlab is stupid

% Save files
file_prefix = input_filename(1:size(input_filename,2)-3);
LMDBTest = strcat(file_prefix, '/LMDB_test');
LMDBTrain = strcat(file_prefix, '/LMDB_train');
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
