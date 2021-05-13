load MNIST_sigmoid_100_50_DenseNet.mat;
exportONNXNetwork(net, 'sigmoid_100_50.onnx');
