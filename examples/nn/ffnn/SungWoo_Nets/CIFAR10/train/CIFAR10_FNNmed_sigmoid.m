close all;
clear;
clc;

normalize = 0;

%Download CIFAR-10 Image Data
%Download the CIFAR-10 data set [3]. This dataset contains 50,000 training images that will be used to train a CNN.
%Download CIFAR-10 data to a temporary directory
cifar10Data = "../dataset/";
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
helperCIFAR10Data.download(url,cifar10Data);

%Load the CIFAR-10 training and test data. 
[trainX_full,trainY_full,testX,testY] = helperCIFAR10Data.load(cifar10Data);

%Each image is a 32x32 RGB image and there are 50,000 training samples.

% % CIFAR-10 has 10 image categories. List the image categories:
% % numImageCategories = 10;
% % categories(trainY_full)
% %Display a few of the training images.
% figure
% thumbnails = trainX_full(:,:,:,1:100);
% montage(thumbnails)


% % Plot 36 smaples of images
% figure                                          % initialize figure
% colormap(gray)                                  % set to grayscale
% for i = 1:36                                    % preview first 36 samples
%     subplot(6,6,i)                              % plot them in 6 x 6 grid
%     digit = reshape(trainX_full(:, i), [28,28]);     % row = 28 x 28 image
%     imagesc(digit)                              % show the image
%     title(num2str(trainY_full(i)))                   % show the label
% end

trainX_full = reshape(trainX_full, [3072 50000]);
trainX = trainX_full(:,1:45000);
trainY = trainY_full(1:45000)';

validX = trainX_full(:,45001:end);
validY = trainY_full(45001:end)';

testX = reshape(testX, [3072 10000]);
testY = testY';

% % Plot 100 smaples of images
% figure                                          % initialize figure
% colormap(gray)                                  % set to grayscale
% for i = 1:100                                    % preview first 150 samples
%     subplot(10,10,i)                              % plot them in 6 x 6 grid
%     digit = reshape(trainX(:, i), [32,32,3]);     % row = 28 x 28 image
%     imagesc(digit)                              % show the image
%     title(trainY(i))                   % show the label
% end

% IM_labels = double(testY)';
% IM_labels = IM_labels -1 ;
% IM_data = testX';
% IM = [IM_labels IM_data];
% writematrix(IM,'cifar_test_full.csv');

% One hot
trainY = categorical(trainY); % Change the data to categorical
validY = categorical(validY); % Change the data to categorical
testY = categorical(testY); % Change the data to categorical


 % Create the image input layer for 32x32x3 CIFAR-10 images.
% [height,width,numChannels, ~] = size(trainX);
% imageSize = [height width numChannels];
% imageSize = [32,32,3];

layers = [ 
    % 'none' — Do not normalize the input data.
    sequenceInputLayer(3072, 'Name', 'input', 'Normalization','none')
    
	flattenLayer('Name', 'flatten')
	
	fullyConnectedLayer(150,'WeightsInitializer', 'glorot', 'Name', 'dense_1')
	sigmoidLayer('Name', 'sigmoid_1')
    
	fullyConnectedLayer(150,'WeightsInitializer', 'glorot', 'Name', 'dense_2')
	sigmoidLayer('Name', 'sigmoid_2')
    
    fullyConnectedLayer(150,'WeightsInitializer', 'glorot', 'Name', 'dense_3')
	sigmoidLayer('Name', 'sigmoid_3')
    
	fullyConnectedLayer(150,'WeightsInitializer', 'glorot', 'Name', 'dense_4')
	sigmoidLayer('Name', 'sigmoid_4')
    
    fullyConnectedLayer(150,'WeightsInitializer', 'glorot', 'Name', 'dense_5')
	sigmoidLayer('Name', 'sigmoid_5')
    
	fullyConnectedLayer(150,'WeightsInitializer', 'glorot', 'Name', 'dense_6')
	sigmoidLayer('Name', 'sigmoid_6')
    
	fullyConnectedLayer(10, 'Name', 'dense_7')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'calssOutput')];

lgraph = layerGraph(layers);
figure
plot(lgraph)


options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 150, ...
    'MaxEpochs', 300, ...
    'MiniBatchSize', 128, ...
    'Shuffle', 'every-epoch', ...
    'Plots','training-progress', ...
    'ValidationData',{validX,validY}, ...
    'ValidationFrequency', 1,...
    'Verbose', 1); %, ...
    

    
net = trainNetwork(trainX,trainY,layers,options);


predY = classify(net,testX/255);
confusion_matrix = confusionmat(testY, predY)
accuracy = sum(predY == testY)/length(testY)

% save ../nets/FNNmed/CIFAR10_FNNmed_sigmoid.mat net;
% 
% load ../nets/FNNmed/CIFAR10_FNNmed_sigmoid.mat;
% predY = classify(net,testX/255);
% confusion_matrix = confusionmat(testY, predY)
% accuracy = sum(predY == testY)/length(testY)

N = 2000; % get 150 images and its labels from the imdsValidation
IM_data = zeros(3072, N);
IM_labels = zeros(N, 1);

n = 1;
for i=1:10000
    if n > N
        break;
    end
    if predY(i) == testY(i)
        IM_data(:, n) = testX(:,n);
        IM_labels(n) = testY(n);
        n = n + 1;
    end
end
IM_labels = IM_labels - 1;

% Plot 150 smaples of images
figure                                          % initialize figure
colormap(gray)                                  % set to grayscale
for i = 1:100                                    % preview first 150 samples
    subplot(10,10,i)                              % plot them in 6 x 6 grid
    digit = reshape(uint8(IM_data(:, i)), [32,32,3]);     % row = 28 x 28 image
    imagesc(digit)                              % show the image
    title(IM_labels(i))                   % show the label
end

IM = [IM_labels IM_data'];
writematrix(IM,'../../CIFAR10/data/CIFAR10_FNNmed_sigmoid_raw.csv');

function images = loadMNISTImages(filename, normalize)
    %loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
    %the raw MNIST images
    fp = fopen(filename, 'rb');
    assert(fp ~= -1, ['Could not open ', filename, '']);
    
    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2051, ['Bad magic number in ', filename, '']);
    
    numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
    numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
    numCols = fread(fp, 1, 'int32', 0, 'ieee-be');
    
    images = fread(fp, inf, 'unsigned char');
    images = reshape(images, numCols, numRows, numImages);
    images = permute(images,[2 1 3]);
    
    fclose(fp);
    
    % Reshape to #pixels x #examples
    images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
    
    % Convert to double and rescale to [0,1]
    if normalize
       images = double(images) / 255.0;
    end
end
    
function labels = loadMNISTLabels(filename)
    %loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
    %the labels for the MNIST images
    fp = fopen(filename, 'rb');
    assert(fp ~= -1, ['Could not open ', filename, '']);
    
    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2049, ['Bad magic number in ', filename, '']);
    
    numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
    labels = fread(fp, inf, 'unsigned char');
    assert(size(labels,1) == numLabels, 'Mismatch in label count');
    
    fclose(fp);
end
