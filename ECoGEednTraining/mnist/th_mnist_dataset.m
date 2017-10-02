function P = th_mnist_dataset(P)
% --------------------------------------------------------------------
% Download and format MNIST
% --------------------------------------------------------------------
% Saves data in train and test LMDBs
% Will only download if does not find LMDBs and downloaded data is not already present
% Will only unpack downloaded data and store in LMDBs if does not find LMDBs
%
% - inputs -
% P                         Structure containing the following fields:
%  .experiment.directory        Top-level directory for all experiment files
%  .dataset.directory           Top-level directory for all dataset files
%  .dataset.trainLmdb           Location of train LMDB, relative to P.dataset.directory
%  .dataset.testLmdb            Location of test LMDB, relative to P.dataset.directory
% 
%
% - outputs -
% P                         Structure containing updated parameters
%
% - internal -
% data                  (row,col,channel,example)
% labels                (example) = class (count from 1)



% --------------------------------
% Set up experiment
% --------------------------------
fprintf('Setting up experiment...\n');

validateattributes(P.dataset.directory,{'char'},{});
validateattributes(P.dataset.trainLmdb,{'char'},{});
validateattributes(P.dataset.testLmdb,{'char'},{});

% Save all experiment parameters.
P = th_log(P, 'dataset');





% --------------------------------
% Prep
% --------------------------------

% --------------------------------
% Download dataset if needed
% --------------------------------
files = {'train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'} ;

if ~exist(P.dataset.directory,'dir')
    mkdir(P.dataset.directory);
end

for fileIdx=1:length(files)
    filename = fullfile(P.dataset.directory, files{fileIdx});
    if ~exist(filename, 'file')
        url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{fileIdx}) ;
        fprintf('Downloading file %s\n', url) ;
        gunzip(url, P.dataset.directory) ;
    end
end

f=fopen(fullfile(P.dataset.directory, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

f=fopen(fullfile(P.dataset.directory, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

f=fopen(fullfile(P.dataset.directory, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(P.dataset.directory, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;


% Train
%dataT = single(reshape(x1,28,28,1,[]))./255; % Rescale
dataT = uint8(reshape(x1,28,28,1,[]));

train.data=zeros(60000,28*28);
train.labels=zeros(60000,10);
for i=1:60000
    temp = squeeze(dataT(:,:,i));
    train.data(i,:) = temp(:);
    train.labels(i,y1(i))=1;
end;

% Test
dataT = uint8(reshape(x2,28,28,1,[]));
test.data=zeros(10000,28*28);
test.labels=zeros(10000,10);
for i=1:10000
    temp = squeeze(dataT(:,:,i));
    test.data(i,:) = temp(:);
    test.labels(i,y2(i))=1;
end;


% ----------------------------------------------------------------
% Format and save train and test data
% ----------------------------------------------------------------
% The recipe will vary with the specifics of each dataset
% For all datasets, end goal is to create a "data" matrix and "labels" matrix for train and test
% data(row,col,channel,example) = data value
%       it is convenient to range data [0 1]
%       or for certain preprocessing steps to allow for negative values and range with std=1
% labels(example) = class label for training example (image)

% Files
trainLmdb = fullfile(P.dataset.directory, P.dataset.trainLmdb);
testLmdb = fullfile(P.dataset.directory, P.dataset.testLmdb);

% Check if done
if exist(trainLmdb,'dir') && exist(testLmdb,'dir');
    return
end

% Select data type of LMDB
dataType = 'uint8'; % Data type can be 'single' or 'uint8'

% --------------------------------------------------
% Write training data to LMDB.
% --------------------------------------------------
fprintf('Creating %s\n', trainLmdb);
mkdir(trainLmdb);

data = reshape( cast(train.data',dataType), 28, 28, 1,[]); 
[~, labels] = max(train.labels,[],2);
labels = int32(labels');

clear write_lmdb; % Close lmdb
write_lmdb(trainLmdb, data, labels, dataType); %write data and give each frame a unique label

% --------------------------------------------------
% Write testing data to LMDB.
% --------------------------------------------------
fprintf('Creating %s\n', testLmdb);
mkdir(testLmdb);

data = reshape( cast(test.data',dataType), 28, 28, 1,[]); 
[~, labels] = max(test.labels,[],2);
labels = int32(labels');

clear write_lmdb; % Close lmdb
write_lmdb(testLmdb, data, labels, dataType); %write data and give each frame a unique label


end 

