% --------------------------------------------------------------------
% Top-level script for an end-to-end EEDN design example using the
% MNIST dataset.
% --------------------------------------------------------------------

%% Set up experiment
%corelet_init; % only needed by build and test, but includes a clear all, so has to be done first.

% Select where to save experiment.
homeDirectory = getenv('HOME');
P.experiment.directory = fullfile(homeDirectory, 'Projects/ECoGAnalysis/ECoGEednTraining/Output/ECoG/BestChannels_1D_Downsampled/Model7/');
P = th_log(P, 'experiment');

% Select which instructions to execute.
%instructions = {'dataset','train','build','test','application'};
instructions = {'dataset','train'};

%% Get dataset
if ismember('dataset', instructions)
    P.dataset.directory = fullfile(homeDirectory, 'Projects/Data/ECoG/ExpandedIsolatedGaussian/BestChannels_1D_Downsampled/');
    P.dataset.trainLmdb = 'ECoG_train';
    P.dataset.testLmdb = 'ECoG_test';
    P = th_log(P, 'dataset');
    P.preprocess.directory = P.dataset.directory;
    P.preprocess.trainLmdb = P.dataset.trainLmdb;
    P.preprocess.testLmdb = P.dataset.testLmdb;
    P = th_log(P, 'preprocess');
end

%% Train network
if ismember('train', instructions)
    P.train.saveNetwork = false;
    P = th_mnist_train(P);
end

%% Build model
if ismember('build', instructions)
    
    loadTrainedNetwork = false;
    if loadTrainedNetwork
        [scriptDirectory,~,~] = fileparts(mfilename('fullpath'));
        networkDirectory = fullfile(scriptDirectory, 'trained_dense_net');
        copyfile(networkDirectory, fullfile(P.experiment.directory, 'network'), 'f');
    end
    P.build.options.singleChip = true;
    P = th_build(P);
end

%% Test model
if ismember('test', instructions)
    P.test.runMode = 'NSCS'; % TN or NSCS
    P.test.nBatches = 1; % Break up test set into this many batches
    P.test.examplesPerBatch = 285; % Number of test examples to run per batch (keep this as large as memory allows)
    P = th_test(P);
end

%% Create application
if ismember('application', instructions)
    P.application.visualizer.address = '127.0.0.1';
    P.application.visualizer.port = 5090;
    P.application.visualizer.totalInstances = 2500;
    P = th_mnist_application(P);
end
