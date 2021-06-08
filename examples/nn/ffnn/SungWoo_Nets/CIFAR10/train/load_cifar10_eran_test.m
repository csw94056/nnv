T = csvread("../data/cifar10_test.csv");
im_data = T(:,2:end);
im_labels = T(:,1);


im = im_data(i,:);
im = reshape(im, [32 32 3]);
im = im';