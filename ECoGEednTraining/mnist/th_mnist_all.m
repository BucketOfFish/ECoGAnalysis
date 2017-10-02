% --------------------------------------------------------------------
% Top-level script for an end-to-end EEDN design example using the
% MNIST dataset.
% --------------------------------------------------------------------

%% Set up experiment
%corelet_init; % only needed by build and test, but includes a clear all, so has to be done first.

% Select where to save experiment.
homeDirectory = getenv('HOME');
P.experiment.directory = fullfile(homeDirectory, 'Projects/ECoGAnalysis/Output/MNIST/experiment');
P = th_log(P, 'experiment');

% Select which instructions to execute.
%instructions = {'dataset','preprocess','train','build','test','application'};
instructions = {'train','build','test','application'};
%instructions = {'application'};

%% Get dataset
if ismember('dataset', instructions)
    P.dataset.directory = fullfile(homeDirectory, 'Projects/ECoGAnalysis/Output/MNIST/data');
    P.dataset.trainLmdb = 'th_lmdb_MNIST_train';
    P.dataset.testLmdb = 'th_lmdb_MNIST_test';

    % If the dataset directory exists we remove it:
    if exist(P.dataset.directory,'file')
        system(['rm -rfv ',P.dataset.directory]);
    end

    P = th_mnist_dataset(P);
end

%% Preprocess dataset
if ismember('preprocess', instructions)
    P.preprocess.directory = P.dataset.directory;
    P.preprocess.trainLmdb = 'th_lmdb_MNIST_train_float';
    P.preprocess.testLmdb = 'th_lmdb_MNIST_test_float';
    P.preprocess.totalTrainInstances = 60000;
    P.preprocess.totalTestInstances = 10000;    
    P = th_mnist_preprocess(P);
end

%% Train network
if ismember('train', instructions)
    P.train.saveNetwork = true;
    P = th_mnist_train(P);
end

%% Build model
if ismember('build', instructions)
    
    loadTrainedNetwork = false;
    if loadTrainedNetwork
        [scriptDirectory,~,~] = fileparts(mfilename('fullpath'));
        networkDirectory = fullfile(scriptDirectory, 'trained_network_1_chip');
        copyfile(networkDirectory, fullfile(P.experiment.directory, 'network'), 'f');
    end
    P.build.options.singleChip = true;
    P = th_build(P);
end

%% Test model
if ismember('test', instructions)
    P.test.runMode = 'TN'; % TN or NSCS
    P.test.nBatches = 4; % Break up test set into this many batches
    P.test.examplesPerBatch = 2500; % Number of test examples to run per batch (keep this as large as memory allows)
    P = th_test(P);
end

%% Create application
if ismember('application', instructions)
    P.application.visualizer.address = '127.0.0.1';
    P.application.visualizer.port = 5090;
    P.application.visualizer.totalInstances = 2500;
    P = th_mnist_application(P);
end
