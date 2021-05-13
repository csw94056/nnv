load MNIST_sigmoid_100_100_DenseNet.mat;
exportONNXNetwork(net, 'sigmoid_100_100.onnx');
