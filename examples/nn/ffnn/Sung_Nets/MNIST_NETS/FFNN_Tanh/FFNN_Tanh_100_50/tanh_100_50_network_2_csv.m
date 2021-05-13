load MNIST_tanh_100_50_DenseNet.mat;
exportONNXNetwork(net, 'tanh_100_50.onnx');
