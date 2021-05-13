load MNIST_sigmoid_100_100_DenseNet.mat net;
load sigmoid_100_100_images.mat;

N = [1,2,3,4,6,7,8,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,...
    26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54];



IM = [IM_labels(N) IM_data(:,N)'];
writematrix(IM,'sigmoid_100_100.csv');