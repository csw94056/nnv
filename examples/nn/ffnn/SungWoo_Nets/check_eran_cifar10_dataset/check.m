clc
clear
close all;
%% load images
image_dir = 'cifar10_test_original.csv';
csv_data = csvread(image_dir);
IM_labels = csv_data(:,1);
IM_data = csv_data(:,2:end)';

% Plot 100 smaples of images
% figure                                          % initialize figure
% for i = 1:1                                    % preview first 150 samples
%     subplot(10,10,i)                              % plot them in 6 x 6 grid
%     digit = permute(reshape(uint8(IM_data(:, i)), [32,32,3]), [2 1 3]);
%     imagesc(digit)                              % show the image
%     title(IM_labels(i))                   % show the label
% end

% figure
for k = 1:100
    I = reshape(IM_data(:,k),3,1024);
    for i = 1:3
       digit(:,:,i) = reshape(I(i,:), [32,32]);
       digit(:,:,i) = digit(:,:,i)';
    end
%    subplot(10,10,k)   
%    imagesc(digit)
%    title(IM_labels(k));
   
   images(k,:,:,:) = digit;
end

digit = [];
for k = 1:100
    digit = reshape(images(k,:,:,:),[32,32,3]);
%     digit = images(k,:,:,:);
    
    for i = 1:3
       digit(:,:,i) = digit(:,:,i)';
       C(i,:) = reshape(digit(:,:,i),1024,1);
    end
    B = reshape(C,[3,1024]);
    IM_check(:,k) = reshape(B,3072,1);
end

%158   112    49   159   111    47   165   116    51   166 %IM_data


% B = reshape(permute(A, [2 1 3]), [27 1])
