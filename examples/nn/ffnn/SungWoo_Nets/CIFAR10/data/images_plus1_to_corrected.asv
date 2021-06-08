close all;
clear;
clc;

%%
% network trained with images: [0 1] -> normalized, 
%                              [0 255] ->  not_normalized
% dataset_ = 'MNIST';
% net_ = 'MNIST_FNNsmall_sigmoid';
dataset_ = 'CIFAR10';
net_ = 'CIFAR10_FNNsmall_tanh';
n_ = 'FNNsmall';
normalized = 0;


close all;
clear;
clc;

%%
% network trained with images: [0 1] -> normalized, 
%                              [0 255] ->  not_normalized
% dataset_ = 'MNIST';
% net_ = 'MNIST_FNNsmall_sigmoid';
dataset_ = 'CIFAR10';
net_ = 'CIFAR10_FNNsmall_tanh';
n_ = 'FNNsmall';
normalized = 0;

image_dir = sprintf('%s_raw_plus1.csv', net_)

csv_data = csvread(image_dir);
IM_labels = csv_data(:,1);
IM_data = csv_data(:,2:end);



IM_labels = IM_labels - 1;

IM = [IM_labels IM_data];
writematrix(IM,'CIFAR10_FNNsmall_tanh_raw.csv');