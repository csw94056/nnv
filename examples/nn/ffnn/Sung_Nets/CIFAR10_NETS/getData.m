%% Classifying the CIFAR-10 dataset using Convolutional Neural Networks
% This example shows how to train a Convolutional Neural Network (CNN) from
% scratch using the dataset CIFAR10.
%
% Data Credit: Krizhevsky, A., & Hinton, G. (2009). Learning multiple 
% layers of features from tiny images.

% Copyright 2016 The MathWorks, Inc.

%% Download the CIFAR-10 dataset
if ~exist('cifar-10-batches-mat','dir')
    cifar10Dataset = 'cifar-10-matlab';
    disp('Downloading 174MB CIFAR-10 dataset...');   
    websave([cifar10Dataset,'.tar.gz'],...
        ['https://www.cs.toronto.edu/~kriz/',cifar10Dataset,'.tar.gz']);
    gunzip([cifar10Dataset,'.tar.gz'])
    delete([cifar10Dataset,'.tar.gz'])
    untar([cifar10Dataset,'.tar'])
    delete([cifar10Dataset,'.tar'])
end    
   
%% Prepare the CIFAR-10 dataset
if ~exist('cifar10Train','dir')
    disp('Saving the Images in folders. This might take some time...');    
    saveCIFAR10AsFolderOfImages('cifar-10-batches-mat', pwd, true);
end

%% Load image CIFAR-10 Training dataset (50000 32x32 colour images in 10 classes)
imsetTrain = imageSet('cifar10Train','recursive');

%% Display Sampling of Image Data
numClasses = size(imsetTrain,2);
imagesPerClass = 10;
imagesInMontage = cell(imagesPerClass,numClasses);
for i = 1:size(imagesInMontage,2)
    imagesInMontage(:,i) = ...
        imsetTrain(i).ImageLocation(randi(imsetTrain(i).Count, 1, ...
        imagesPerClass));
end

montage({imagesInMontage{:}},'Size',[numClasses,imagesPerClass]);
title('Sample of Training Data (Credit:Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.)')

%% Prepare the data for Training
% Read all images and store them in a 4D uint8 input array for training,
% with its corresponding class

trainNames = {imsetTrain.Description};
XTrain = zeros(32,32,3,sum([imsetTrain.Count]),'uint8');
TTrain = categorical(discretize((1:sum([imsetTrain.Count]))',...
    [0,cumsum([imsetTrain.Count])],'categorical',trainNames));

j = 0;
tic;
for c = 1:length(imsetTrain)
    for i = 1:imsetTrain(c).Count
        XTrain(:,:,:,i+j) = read(imsetTrain(c),i);
    end
    j = j + imsetTrain(c).Count;
end
toc;