load MNIST_sigmoid_100_50_DenseNet.mat net;
load sigmoid_100_50_images.mat;

N = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,19,20,22,23,24,25,26,27,29,...
    31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,48,49,50,51,52,53,54,55,56,57];

IM = [IM_labels(N) IM_data(:,N)'];
writematrix(IM,'sigmoid_100_50.csv');



figure                                          % initialize figure
colormap(gray)                                  % set to grayscale
for i = 1:50                                    % preview first 36 samples
    subplot(5,10,i)                              % plot them in 6 x 6 grid
    digit = reshape(IM_data(:, i), [28,28]);     % row = 28 x 28 image
    imagesc(digit)                              % show the image
    title(num2str(IM_labels(i)))                   % show the label
end
