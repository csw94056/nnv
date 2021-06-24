% The MNIST data set consists of 70,000 handwritten digits split into 
% training and test partitions of 60,000 and 10,000 images, respectively. 
% Each image is 28-by-28 pixels and has an associated label denoting which 
% digit the image represents (0–9).
close all;
clear;
clc;

% Load MNIST Data
normalize = 0;
trainX_full = loadMNISTImages('../../MNIST/dataset/train-images.idx3-ubyte', normalize);
trainY_full = loadMNISTLabels('../../MNIST/dataset/train-labels.idx1-ubyte');
testX = loadMNISTImages('../../MNIST/dataset/t10k-images.idx3-ubyte', normalize);
testY = loadMNISTLabels('../../MNIST/dataset/t10k-labels.idx1-ubyte');

% % Plot 36 smaples of images
% figure                                          % initialize figure
% colormap(gray)                                  % set to grayscale
% for i = 1:36                                    % preview first 36 samples
%     subplot(6,6,i)                              % plot them in 6 x 6 grid
%     digit = reshape(trainX_full(:, i), [28,28]);     % row = 28 x 28 image
%     imagesc(digit)                              % show the image
%     title(num2str(trainY_full(i)))                   % show the label
% end

trainX = trainX_full(:,1:50000);
trainY = trainY_full(1:50000,:)';

validX = trainX_full(:,50001:60000);
validY = trainY_full(50001:60000,:)';

% One hot
trainY = categorical(trainY); % Change the data to categorical
validY = categorical(validY); % Change the data to categorical
testY = categorical(testY); % Change the data to categorical


% layers = [ 
%     % 'none' — Do not normalize the input data.
%     sequenceInputLayer(784, 'Name', 'input', 'Normalization','none')
% 	flattenLayer('Name', 'flatten')
% 	
% 	fullyConnectedLayer(150,'WeightsInitializer', 'glorot', 'Name', 'dense_1')
% 	tanhLayer('Name', 'tanh_1')
% 	
% 	fullyConnectedLayer(150,'WeightsInitializer', 'glorot', 'Name', 'dense_2')
% 	tanhLayer('Name', 'tanh_2')
%     
%     fullyConnectedLayer(150,'WeightsInitializer', 'glorot', 'Name', 'dense_3')
% 	tanhLayer('Name', 'tanh_3')
% 	
% 	fullyConnectedLayer(150,'WeightsInitializer', 'glorot', 'Name', 'dense_4')
% 	tanhLayer('Name', 'tanh_4')
%     
%     fullyConnectedLayer(150,'WeightsInitializer', 'glorot', 'Name', 'dense_5')
% 	tanhLayer('Name', 'tanh_5')
%     
%     fullyConnectedLayer(150,'WeightsInitializer', 'glorot', 'Name', 'dense_6')
% 	tanhLayer('Name', 'tanh_6')
%     
% 	fullyConnectedLayer(10, 'WeightsInitializer', 'glorot', 'Name', 'dense_7')
%     softmaxLayer('Name', 'softmax')
%     classificationLayer('Name', 'calssOutput')];
% 
% lgraph = layerGraph(layers);
% figure
% plot(lgraph)
% 
% 
% options = trainingOptions('adam', ...
%     'InitialLearnRate', 0.01, ...
%     'LearnRateSchedule', 'piecewise', ...
%     'LearnRateDropFactor', 0.1, ...
%     'LearnRateDropPeriod', 200, ... %200
%     'MaxEpochs', 270,... %270, ...
%     'MiniBatchSize', 128, ...
%     'Verbose', 1, ...
%     'Shuffle', 'every-epoch', ...
%     'Plots','training-progress', ...
%     'ValidationData',{validX,validY}, ...
%     'ValidationFrequency', 1)
%     
% net = trainNetwork(trainX,trainY,layers,options);
% 
% 
% predY = classify(net,testX)';
% confusion_matrix = confusionmat(testY, predY)
% accuracy = sum(predY == testY)/length(testY)
% 
% save ../../MNIST/nets/FNNbig/MNIST_FNNbig_tanh.mat net;

load ../../MNIST/nets/FNNbig/MNIST_FNNmed_tanh.mat;
predY = classify(net,testX/255)';
confusion_matrix = confusionmat(testY, predY)
accuracy = sum(predY == testY)/length(testY)
% 
% classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
% net = importONNXNetwork('ffnnTANH__PGDK_w_0.1_6_500.onnx','OutputLayerType','classification','Classes',classes);
% digit = reshape(testX, [28 28 1 10000]);
% 
% predY = classify(net,digit);
% accuracy = sum(predY == testY)/length(testY)

N = 10000; % get 150 images and its labels from the imdsValidation
IM_data = zeros(784, N);
IM_labels = zeros(N, 1);

n = 1;
for i=1:10000
    if n > N
        break;
    end
    if predY(i) == testY(i)
        IM_data(:, n) = testX(:,n);
        IM_labels(n) = str2num(char(testY(n)));
        n = n + 1;
    end
end

% % Plot 150 smaples of images
% figure                                          % initialize figure
% colormap(gray)                                  % set to grayscale
% for i = 1:150                                    % preview first 150 samples
%     subplot(15,10,i)                              % plot them in 6 x 6 grid
%     digit = reshape(IM_data(:, i), [28,28]);     % row = 28 x 28 image
%     imagesc(digit)                              % show the image
%     title(IM_labels(i))                   % show the label
% end

IM = [IM_labels(1:n) IM_data(:,1:n)'];
writematrix(IM,'../../MNIST/data/MNIST_FNNmed_tanh_raw.csv');

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
