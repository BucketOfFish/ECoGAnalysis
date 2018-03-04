function P = th_mnist_train(P)
% --------------------------------------------------------------------
% Compose and train a TNCN network
% --------------------------------------------------------------------
%
% - inputs -
% P                         Structure containing the following fields:
%  .experiment.directory        Top-level directory for all experiment files
%  .preprocess.directory        Top-level directory for all preprocessed LMDB
%  .preprocess.trainLmdb        Location of preprocessed train LMDB (single), relative to P.dataset.directory
%  .preprocess.testLmdb         Location of preprocessed test LMDB (single), relative to P.dataset.directory
%  .train.saveNetwork           If 'true', save network files to P.train.directory
%                                   - Throws an error if 'true' and P.train.directory already exists
% 
%
% - outputs -
% P                         Structure containing updated parameters
%  .train.directory             Directory where training files are saved
%                                   - Always a subdirectory of P.experiment.directory
%                                   - Always named 'network'
%  .train.networkFile           File containing all network parameters and training results
%                                   - Necessary and sufficient to build a TNCN model
%                                   - Always named 'networkForCorelet.mat'



% ----------------------------------------------------------------------
% Prep
% ----------------------------------------------------------------------
clc;
close all;

% ----------------------------------------------------------------------
% Data
% ----------------------------------------------------------------------

% Parameters
if P.train.saveNetwork
    
    % Save training parameters
    P.train.directory = 'network';
    P.train.networkFile = 'networkForCorelet.mat';
    P = th_log(P, 'train');
    
    % Save trained network
    saveDirectory = fullfile(P.experiment.directory, P.train.directory);
else
    saveDirectory = [];
end
trainLmdb = fullfile(P.dataset.directory, P.dataset.trainLmdb);
testLmdb = fullfile(P.dataset.directory, P.dataset.testLmdb);

% ----------------------------------------------------------------------
% Network
% ----------------------------------------------------------------------

% ----------------------------------------
% Data Layer
% ---------------------------------------
N{1} = th_layerdata({'trainFile',trainLmdb,'testFile',testLmdb,'nClasses',57,'batchSize',100,'shuffleTrain',true});

% ----------------------------------------
% Neuron Layers - 1D
% ----------------------------------------

%dropoutRate = 0.25; learningRate = 0.3;
%N{end+1} = th_layerconv_bin(N{end},{'nFeatures',16,'patchSize',[1,2],'nGroups',1,'pad',0,'stride',2,'transduction',true,'weightInitMethod','gauss3200','hysteresis',0}); % Transduction layer
%N{end+1} = th_layerdrop(N{end},{'rate',dropoutRate}); 
%N{end+1} = th_layerconv_bin(N{end},{'nFeatures',32,'patchSize',[1,3],'nGroups',1,'pad',0,'stride',3});
%N{end+1} = th_layerdrop(N{end},{'rate',dropoutRate}); 
%N{end+1} = th_layerconv_bin(N{end},{'nFeatures',32,'patchSize',[1,1],'nGroups',1,'pad',0,'stride',1});
%N{end+1} = th_layerdrop(N{end},{'rate',dropoutRate}); 
%N{end+1} = th_layerconv_bin(N{end},{'nFeatures',32,'patchSize',[1,3],'nGroups',1,'pad',0,'stride',3});
%N{end+1} = th_layerdrop(N{end},{'rate',dropoutRate}); 
%N{end+1} = th_layerconv_bin(N{end},{'nFeatures',16,'patchSize',[1,1],'nGroups',1,'pad',0,'stride',1});
%N{end+1} = th_layerdrop(N{end},{'rate',dropoutRate}); 
%N{end+1} = th_layerconv_bin(N{end},{'nFeatures',64,'patchSize',[1,7],'nGroups',1,'pad',0,'stride',7});
%N{end+1} = th_layerdrop(N{end},{'rate',dropoutRate}); 
%N{end+1} = th_layerconv_bin(N{end},{'nFeatures',64,'patchSize',[1,1],'nGroups',1,'pad',0,'stride',1});
%N{end+1} = th_layerdrop(N{end},{'rate',dropoutRate}); 
%N{end+1} = th_layerconv_bin(N{end},{'nFeatures',64,'patchSize',[1,1],'nGroups',1,'pad',0,'stride',1});
%%N{end+1} = th_layerdrop(N{end},{'rate',dropoutRate}); 
%%N{end+1} = th_layerconv_bin(N{end},{'nFeatures',8,'patchSize',[1,1],'nGroups',1,'pad',0,'stride',1});
%%N{end+1} = th_layerdrop(N{end},{'rate',dropoutRate}); 
%%N{end+1} = th_layerconv_bin(N{end},{'nFeatures',64,'patchSize',[1,10],'nGroups',1,'pad',0,'stride',1});

% ----------------------------------------
% Neuron Layers - 2D
% ----------------------------------------

dropoutRate = 0.5; learningRate = 0.3;
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',16,'patchSize',[1,10],'nGroups',1,'pad',0,'stride',1,'transduction',true,'weightInitMethod','gauss3200','hysteresis',0}); % Transduction layer
N{end+1} = th_layerdrop(N{end},{'rate',dropoutRate}); 
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',32,'patchSize',[6,1],'nGroups',1,'pad',0,'stride',3});
N{end+1} = th_layerdrop(N{end},{'rate',dropoutRate}); 
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',32,'patchSize',[1,1],'nGroups',1,'pad',0,'stride',1});
N{end+1} = th_layerdrop(N{end},{'rate',dropoutRate}); 
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',32,'patchSize',[3,1],'nGroups',1,'pad',0,'stride',3});
N{end+1} = th_layerdrop(N{end},{'rate',dropoutRate}); 
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',16,'patchSize',[1,1],'nGroups',1,'pad',0,'stride',1});
N{end+1} = th_layerdrop(N{end},{'rate',dropoutRate}); 
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',64,'patchSize',[7,1],'nGroups',1,'pad',0,'stride',7});
N{end+1} = th_layerdrop(N{end},{'rate',dropoutRate}); 
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',64,'patchSize',[1,1],'nGroups',1,'pad',0,'stride',1});
N{end+1} = th_layerdrop(N{end},{'rate',dropoutRate}); 
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',128,'patchSize',[1,1],'nGroups',1,'pad',0,'stride',1});

% ----------------------------------------
% Predict Layer
% ----------------------------------------
N{end+1} = th_layerpredict(N{end},{'predictionRange',10});

% ----------------------------------------
% Loss Layer
% ----------------------------------------
N{end+1} = th_layerloss(N{end}, {'lossFunction', 'softmax', 'topk', 1});
%N{end+1} = th_layerloss(N{end});

% Default number of training iterations. These must be the numbers for a
% complete training run. For a shorter run, set these values in the *_all.m
% script to smaller values
P.train=propDefault(P.train,'firstTrainIter',200000); % for long test use 200000
P.train=propDefault(P.train,'otherTrainIter',50000); % for long test use 50000
P.train=propDefault(P.train,'testIter',16); % for long test use 79
P.train=propDefault(P.train,'pauseOnCreation',false); % for long test use true

%P.train.firstTrainIter = 20000;
%P.train.testIter = 16;
%P.train.otherTrainIter = 5000;

% ----------------------------------------
% Net (Solver + Result Tracker)
% ----------------------------------------
T = th_net_bin({'trainIters',P.train.firstTrainIter,'trueNorthTestPeriod',100000,'saveDirectory',saveDirectory,'testIters',P.train.testIter,'pauseOnCreation',P.train.pauseOnCreation});
%T = th_net_bin({'trainIters',P.train.firstTrainIter,'trueNorthTestPeriod',min(1000,P.train.firstTrainIter/2),'saveDirectory',saveDirectory,'testIters',P.train.testIter,'pauseOnCreation',P.train.pauseOnCreation});

% ----------------------------------------
% Train
% ----------------------------------------
% Learn
N{1}.setAll({'learnRate', learningRate});
%N{1}.setAll({'weightDecay',0});
T.train(N);

% ----------------------------------------
% Fine Tune
% ----------------------------------------
% Learning rate decay
T.trainIters = P.train.otherTrainIter;
N{1}.setAll({'learnRate',N{1}.learningRate .* 0.5});
T.train(N);
N{1}.setAll({'learnRate',N{1}.learningRate .* 0.5});
T.train(N);

fprintf('th_mnist_train:  Done.\n');

% Plot
%figure
plotFile = P.experiment.directory + P.train.directory + P.train.networkFile;
plotEednLayersFile(1, plotFile)

end
