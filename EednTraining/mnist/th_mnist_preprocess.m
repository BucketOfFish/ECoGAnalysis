function P = th_mnist_preprocess(P)
% --------------------------------------------------------------------
% Preprocesses original SVHN data using contrast normalization
% and create lmdb file with data in the range [-1..1].
% --------------------------------------------------------------------
%
% - inputs -
% P                         Structure containing the following fields:
%  .experiment.directory        Top-level directory for all experiment files
%  .experiment.parameterFile    Parameter log file in P.experiment.directory
%  .dataset.directory           Top-level directory for all dataset files
%  .dataset.trainLmdb           Location of original train LMDB (uint8), relative to P.dataset.directory
%  .dataset.testLmdb            Location of original test LMDB (uint8), relative to P.dataset.directory
%  .preprocess.directory        Top-level directory for all preprocessed LMDB
%  .preprocess.trainLmdb        Location of preprocessed train LMDB (single), relative to P.dataset.directory
%  .preprocess.testLmdb         Location of preprocessed test LMDB (single), relative to P.dataset.directory
%  .preprocess.totalTrainInstances  Number of instances from P.dataset.trainLmdb to include in P.preprocess.trainLmdb
%  .preprocess.totalTestInstances   Number of instances from P.dataset.testLmdb to include in P.preprocess.testLmdb
%
% - outputs -
% P                         Structure containing updated parameters


% Status
fprintf('Contrast Normalization...\n');

% Check inputs
validateattributes(P.experiment.directory,{'char'},{});
validateattributes(P.experiment.parameterFile,{'char'},{});
validateattributes(P.dataset.directory,{'char'},{});
validateattributes(P.dataset.trainLmdb,{'char'},{});
validateattributes(P.dataset.testLmdb,{'char'},{});
validateattributes(P.preprocess.trainLmdb,{'char'},{});
validateattributes(P.preprocess.testLmdb,{'char'},{});
validateattributes(P.preprocess.totalTestInstances,{'numeric'},{});
validateattributes(P.preprocess.totalTrainInstances,{'numeric'},{});

% Save preprocessing parameters for later stages (training, testing...)
P = th_log(P, 'preprocess');

% Check if need to run
trainLmdbPre = fullfile(P.preprocess.directory, P.preprocess.trainLmdb);
testLmdbPre = fullfile(P.preprocess.directory, P.preprocess.testLmdb);
if exist(trainLmdbPre,'dir') && exist(testLmdbPre,'dir');    
    return
end

%% Convert test data

mkdir(testLmdbPre);

imageSize = [32 32 3];
scale = single(1.0/255.0);

testFile = fullfile(P.dataset.directory, P.dataset.testLmdb);
disp(['Loading ' testFile]);
firstSample = uint32(1);
dataType = 'uint8';
clear read_lmdb;
[X,labels,~]=read_lmdb(testFile,uint32(P.preprocess.totalTestInstances), ...
    firstSample, dataType);

% Cast from uint8 to float.
X=single(X);

% Rescale to [0..1].
data = scale*single(X);

% Save to LMDB and .mat.
disp(['Writing ' testLmdbPre]);
save_as_lmdb(testLmdbPre, data, labels);

%% Convert training data.
mkdir(trainLmdbPre);

trainFile = fullfile(P.dataset.directory, P.dataset.trainLmdb);
disp(['Loading ' trainFile]);
firstSample = uint32(1);
dataType = 'uint8';
clear read_lmdb;
[X,labels,~]=read_lmdb(trainFile,uint32(P.preprocess.totalTrainInstances), ...
    firstSample, dataType);

% Cast from uint8 to float.
X=single(X);

% Randomly shuffle data order.
perm = randperm(size(X,4));
X = X(:,:,:,perm);
labels=labels(perm);        

% Rescale to [0..1].
data=scale*single(X);

% Save to LMDB and .mat.
disp(['Writing ' trainLmdbPre]);
save_as_lmdb(trainLmdbPre, data, labels);

disp('Done Preprocessing');

end

function save_as_lmdb(name, data, labels)
    dataType = 'single';

    labels = int32(labels);
    clear write_lmdb; % Close LMDB
    
    % Create LMDB dir if needed
    if ~exist(name, 'dir')
        mkdir(name);
    end
    
    % Write data and give each frame a unique label
    write_lmdb(name, data, labels, dataType);
end
