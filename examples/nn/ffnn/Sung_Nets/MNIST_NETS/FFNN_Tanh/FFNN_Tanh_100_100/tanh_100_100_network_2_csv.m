load MNIST_tanh_100_100_DenseNet.mat;
exportONNXNetwork(net, 'tanh_100_100.onnx');
