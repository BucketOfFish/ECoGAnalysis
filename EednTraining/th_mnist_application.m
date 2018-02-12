function P = th_mnist_application(P)
% --------------------------------------------------------------------
% Create standalone application including visualizer and hardware
% --------------------------------------------------------------------
%
% - inputs -
% P                             Structure containing the following fields:
%  .experiment.directory                Top-level directory for all experiment files
%  .experiment.parameterFile            Parameter log file in P.experiment.directory
%  .dataset.directory                   Top-level directory for all dataset files
%  .dataset.testLmdb                    Location of test LMDB, relative to P.dataset.directory
%  .build.codeWindow                    Number of coding ticks per frame
%  .build.resetWindow                   Number of reset ticks per frame
%  .test.examplesPerBatch               Number of test instances that were run per batch
%  .application.visualizer.address      Network address of visualizer
%  .application.visualizer.port         Network port (UDP) of visualizer
%
% 
%
% - outputs -
% P                             Structure containing updated parameters
%  .application.directory           Directory where application files are saved
%                                       - Always a subdirectory of P.experiment.directory
%                                       - Always named 'application'
%  .application.name                Prefix for application files (<name>.sh, <name>.conf)
%
%  .application.visualizer.categoryIconDirectory	Location of category icon files, relative to P.application.directory
%  .application.visualizer.categoryIcons            List of category icon files
%
%  .application.visualizer.confidenceBias
%
%  .application.visualizer.windowSize
%  .application.visualizer.imageWidth
%  .application.visualizer.imageAspectRatio
%  .application.visualizer.imageOffsetX
%  .application.visualizer.imageOffsetY
%  .application.visualizer.iconWidth
%  .application.visualizer.iconAspectRatio
%  .application.visualizer.iconOutline
%  .application.visualizer.iconSelectedOutline
%  .application.visualizer.iconArrayWidth
%  .application.visualizer.iconArrayOffsetX
%  .application.visualizer.iconArrayOffsetY
%  .application.visualizer.iconArrayStrideX
%  .application.visualizer.iconArrayStrideY
%  .application.visualizer.confidenceWidth
%  .application.visualizer.confidenceHeight
%  .application.visualizer.confidenceOffsetY
%
%  .application.visualizer.codeWindow
%  .application.visualizer.resetWindow
%  .application.visualizer.inputLatency
%  .application.visualizer.outputLatency
%  .application.visualizer.totalCategories          Number of classification categories
%  .application.visualizer.modelDirectory           Location of model and spike files, relative to P.application.directory
%  .application.visualizer.outputMapDirectory       Location of output connector map (same as P.application.visualizer.modelDirectory)
%  .application.visualizer.outputMapFile            Name of output connector map (in P.application.visualizer.outputMapDirectory)
%  .application.visualizer.neuronCategoryDirectory  Location of neuron category file (same as P.application.visualizer.modelDirectory)
%  .application.visualizer.neuronCategoryFile       Name of text file mapping output neurons to categories (in P.application.visualizer.neuronCategoryDirectory)
%  .application.visualizer.lmdbDirectory            Location of local copy of original test LMDB, relative to P.application.directory
%  .application.visualizer.launchScript             Name of visualizer launch script = [P.application.name,'.sh']
%
%  .application.hardware.modelFile              Name of model file (in P.application.visualizer.modelDirectory)
%  .application.hardware.inputSpikeFile         Name of input spike file (in P.application.visualizer.modelDirectory)
%  .application.hardware.inputMapFile           Name of input connector map (in P.application.visualizer.modelDirectory)
%
%  .application.hardware.tickCount          Number of ticks per loop (must be a multiple of 16)
%  .application.hardware.tickPeriod         Milliseconds per tick
%  .application.hardware.configFile         Name of hardware configuration file = [P.application.name,'.conf'];


P.application.name = 'MnistApp';
visualizer = P.application.visualizer;

% Location of test data - use the original images, not the preprocessed data
visualizer.sourceLmdbDirectory = fullfile(P.dataset.directory, P.dataset.testLmdb);


% Icon file for each category (.jpg or .png)
% the /images folder is under the same directory as this function. 
visualizer.sourceIconDirectory = [fileparts(mfilename('fullpath')) '/images'];
totalCategories = 10;
visualizer.categoryIcons = cell(1,totalCategories);
for category = 1:totalCategories
    visualizer.categoryIcons{category} = sprintf('mnist%d.png',category-1);
end

% Confidence bar threshold
visualizer.confidenceBias = 0.90;

% Visualizer layout
visualizer.windowSize = [480 272];
visualizer.imageWidth = 0.228;
visualizer.imageAspectRatio = 1.0;
visualizer.imageOffsetX = 0.075;
visualizer.imageOffsetY = 0.3;
visualizer.iconWidth = 0.1;
visualizer.iconOutline = 0.002;
visualizer.iconSelectedOutline = 0.012;
visualizer.iconAspectRatio = 1.0;
visualizer.iconArrayWidth = 5;
visualizer.iconArrayOffsetX = 0.38;
visualizer.iconArrayOffsetY = 0.24;
visualizer.iconArrayStrideX = 1.2;
visualizer.iconArrayStrideY = 2.0;
visualizer.confidenceWidth = 0.09;
visualizer.confidenceHeight = 0.03;
visualizer.confidenceOffsetY = 0.04;

% Configure visualizer
P.application.visualizer = visualizer;
P = th_application(P);
