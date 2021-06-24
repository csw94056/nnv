load CIFAR10_FNNsmall_tanh.mat

W1 = net.Layers(3).Weights;
b1 = net.Layers(3).Bias;

W2 = net.Layers(5).Weights;
b2 = net.Layers(5).Bias;

W3 = net.Layers(7).Weights;
b3 = net.Layers(7).Bias;

save CIFAR10_FNNsmall_tanh_matirx.mat W1 b1 W2 b2 W3 b3