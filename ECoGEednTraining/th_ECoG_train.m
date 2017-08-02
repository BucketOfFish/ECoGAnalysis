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
trainLmdb = fullfile(P.preprocess.directory, P.preprocess.trainLmdb);
testLmdb = fullfile(P.preprocess.directory, P.preprocess.testLmdb);

% ----------------------------------------------------------------------
% Network
% ----------------------------------------------------------------------

% ----------------------------------------
% Data Layer
% ---------------------------------------
N{1} = th_layerdata({'trainFile',trainLmdb,'testFile',testLmdb,'nClasses',10,'dataScale',255,'dataAdd',-128,'floorMode',true,'batchSize',128,'shuffleTrain',true});


% ----------------------------------------
% Neuron Layers
% ----------------------------------------

% Preprocessing
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',12,'patchSize',[3 3],'nGroups',1,'pad',1,'transduction',true,'weightInitMethod','gauss3200','hysteresis',0}); % Topo

% Set 1
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',252, 'patchSize',[4 4],'nGroups',2,'pad',1,'stride',2}); % Pool
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',256, 'patchSize',[1 1],'nGroups',2}); % NIN
% N{end+1} = th_layerconv_bin(N{end},{'nGroups',2,'patchSize',[1 1],'nFeatures',256}); % NIN

% Set 2
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',256,'patchSize',[2 2],'nGroups',8,'pad',0,'stride',2}); % Pool
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',512,'patchSize',[3 3],'nGroups',32,'pad',1}); % Topo
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',512,'patchSize',[1 1],'nGroups',4}); % NIN
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',512,'patchSize',[1 1],'nGroups',4}); % NIN
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',512,'patchSize',[1 1],'nGroups',4}); % NIN

% Set 3
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',512,'patchSize',[2 2],'nGroups',16,'pad',0,'stride',2}); % Pool
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',1024,'patchSize',[3 3],'nGroups',64,'pad',1}); % Topo
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',1024,'patchSize',[1 1],'nGroups',8}); % NIN

% Set 4
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',1024,'patchSize',[2 2],'nGroups',32,'pad',0,'stride',2}); % Pool
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',1024,'patchSize',[1 1],'nGroups',8}); % NIN
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',1024,'patchSize',[1 1],'nGroups',8}); % NIN
%N{end+1} = th_layerdrop(N{end},{'rate',0.5}); 
N{end+1} = th_layerconv_bin(N{end},{'nFeatures',2040,'patchSize',[1 1],'nGroups',8}); % NIN

% ----------------------------------------
% Predict Layer
% ----------------------------------------
N{end+1} = th_layerpredict(N{end},{'predictionRange',10});


% ----------------------------------------
% Loss Layer
% ----------------------------------------
N{end+1} = th_layerloss(N{end});

% Default number of training iterations. These must be the numbers for a
% complete training run. For a shorter run, set these values in the *_all.m
% script to smaller values
P.train=propDefault(P.train,'firstTrainIter',2); % for long test use 200000
P.train=propDefault(P.train,'otherTrainIter',500); % for long test use 50000
P.train=propDefault(P.train,'testIter',16); % for long test use 79
P.train=propDefault(P.train,'pauseOnCreation',false); % for long test use true

% ----------------------------------------
% Net (Solver + Result Tracker)
% ----------------------------------------
T = th_net_bin({'trainIters',P.train.firstTrainIter,'trueNorthTestPeriod',min(1000,P.train.firstTrainIter/2),'saveDirectory',saveDirectory,'testIters',P.train.testIter,'pauseOnCreation',P.train.pauseOnCreation});

% ----------------------------------------
% Train
% ----------------------------------------
% Learn
N{1}.setAll({'learnRate',20});
%N{1}.setAll({'weightDecay',0});
T.train(N);

% ----------------------------------------
% Fine Tune
% ----------------------------------------
% Learning rate decay
%T.trainIters = P.train.otherTrainIter;
%N{1}.setAll({'learnRate',N{2}.learnRate .* 0.1});
%T.train(N);
%N{1}.setAll({'learnRate',N{2}.learnRate .* 0.1});
%T.train(N);

fprintf('th_mnist_train:  Done.\n');

end
