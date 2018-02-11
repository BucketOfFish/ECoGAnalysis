clc;

close all;

clear;



TRAINING_DATA = fullfile('/home/pras/Failures/data/lmdb', 'sur_ras_train_lmdb');

TEST_DATA = fullfile('/home/pras/Failures/data/lmdb', 'sur_ras_test_lmdb');

MODEL_PATH = fullfile('/home/pras/Failures', 'model', 'dnn_model.out')



% Remove the save directory

if exist(MODEL_PATH, 'file')
	
	rmdir '../model/dnn_model.out' 's'

end



% Data layer

N{1} = th_layerdata({'trainFile', TRAINING_DATA, 'testFile', TEST_DATA, ...

			'nClasses', 5, 'batchSize', 128, 'dataScale', 1, ...
			'dataAdd', 0, 'floorMode', true, ...

			'lmdbDataType', 'uint8', 'shuffleTrain', true, ...

			'mirror', false});



% Transduction 
layer
N{end+1} = th_layerconv_bin(N{end}, {'patchSize', [1,1], 'nGroups', 1, ...

						'pad', 0, 'stride', 1, ...

						'nFeatures', 30, ...

						'weightInitMethod', 'gauss100', ...

						'learnRate', 0.1, 'momentum', 0.9, ...

						'hysteresis', 0, ...

						'transduction', true});



% Hidden Layer 1

N{end+1} = th_layerconv_bin(N{end}, {'patchSize', [1,1], 'nGroups', 1, ...

					'pad', 0, 'stride', 1, ...

					'nFeatures', 20, ...

					'weightInitMethod', 'gauss100', ...

					'learnRate', 1, 'momentum', 0.9, ...

					'hysteresis', 0.1, ...

					'transduction', false});



% Hidden Layer 2

%N{end+1} = th_layerconv_bin(N{end}, {'patchSize', [1,1], 'nGroups', 1, ...

%					'pad', 0, 'stride', 1, ...

%					'nFeatures', 10, ...

%					'weightInitMethod', 'gauss100', ...

%					'learnRate', 1, 'momentum', 0.9, ...

%					'hysteresis', 0.1, ...

%					'transduction', false});



% Prediction Layer

N{end+1} = th_layerpredict(N{end}, {'predictFunction', 'halfprob', ...

					'predictionRange', 10});



% Loss Layer

N{end+1} = th_layerloss(N{end}, {'lossFunction', 'softmax', ...

					'topk', 1});



% Net Scaffolding Object
T = th_net_bin({'testIters', 10, 'trainIters', 100, 'testPeriod', 10, ...

					'trueNorthTestPeriod', 10000, 'savePeriod', 100, ...

					'saveDirectory', MODEL_PATH, 'pauseOnCreation', false});



% Train

T.train(N)
