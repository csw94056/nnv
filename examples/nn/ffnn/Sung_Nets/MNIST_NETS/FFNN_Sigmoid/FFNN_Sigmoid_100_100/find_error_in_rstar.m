close all;
clear;
clc;

load not_normalized/inputRStar12.mat;
load not_normalized/inputStar12.mat;
load not_normalized/inputAbsDom12.mat;

load MNIST_sigmoid_100_100_DenseNet.mat net
L1 = LayerS(net.Layers(3).Weights, net.Layers(3).Bias, 'logsig');
L2 = LayerS(net.Layers(5).Weights, net.Layers(5).Bias, 'logsig');
L3 = LayerS(net.Layers(7).Weights, net.Layers(7).Bias, 'purelin');
nnv_net = FFNNS([L1 L2 L3]);

N = 50; 
numCores = 1;
reachMethod = 'rstar-absdom-two';

load images.mat;
labels = IM_labels;

% for i=1:N
%     i
%     [r12, rb12, cE12, cands12, vt12] = nnv_net.evaluateRBN(RS_eps_12(i), labels(i)+1, reachMethod, numCores, 0, 0, 'glpk');
%     epsilon12 = [1.2];
%     verify_time12 = [sum(vt12)];
%     safe12 = [sum(rb12==1)];
%     unsafe12 = [sum(rb12 == 0)];
%     unknown12 = [sum(rb12 == 2)];
%     T12 = table(epsilon12, safe12, unsafe12, unknown12, verify_time12)
%     fprintf('total time rstar (eps=1.2): %f ',verify_time12);
% end


% 21 th image causes ill-conditioned matrix
i=21;
[r12, rb12, cE12, cands12, vt12] = nnv_net.evaluateRBN(RS_eps_12(i), labels(i)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon12 = [1.2];
verify_time12 = [sum(vt12)];
safe12 = [sum(rb12==1)];
unsafe12 = [sum(rb12 == 0)];
unknown12 = [sum(rb12 == 2)];
T12 = table(epsilon12, safe12, unsafe12, unknown12, verify_time12)
fprintf('total time rstar (eps=1.2): %f ',verify_time12);