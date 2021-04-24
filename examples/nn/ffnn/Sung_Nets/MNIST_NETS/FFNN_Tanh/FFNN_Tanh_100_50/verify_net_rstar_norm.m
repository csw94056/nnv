close all;
clear;
clc;

load tanh_100_50_normalized/inputRStar_norm0001.mat;
load tanh_100_50_normalized/inputRStar_norm0002.mat;
load tanh_100_50_normalized/inputRStar_norm0003.mat;
load tanh_100_50_normalized/inputRStar_norm0004.mat;
load tanh_100_50_normalized/inputRStar_norm0005.mat;
load tanh_100_50_normalized/inputRStar_norm0006.mat;
load tanh_100_50_normalized/inputRStar_norm0007.mat;
load tanh_100_50_normalized/inputRStar_norm0008.mat;
load tanh_100_50_normalized/inputRStar_norm0009.mat;
load tanh_100_50_normalized/inputRStar_norm0010.mat;

load MNIST_tanh_100_50_normalized_DenseNet.mat net;
L1 = LayerS(net.Layers(3).Weights, net.Layers(3).Bias, 'tansig');
L2 = LayerS(net.Layers(5).Weights, net.Layers(5).Bias, 'tansig');
L3 = LayerS(net.Layers(7).Weights, net.Layers(7).Bias, 'purelin');
nnv_net = FFNNS([L1 L2 L3]);

load tanh_100_50_images_normalized.mat;
labels = IM_labels;

% figure                                          % initialize figure
% colormap(gray)                                  % set to grayscale
% for i = 1:50                                    % preview first 50 samples
%     subplot(5,10,i);                              % plot them in 6 x 6 grid
%     digit = reshape(RS_eps_norm_005(i).lb{1}, [28,28]);     % row = 28 x 28 image
%     imagesc(digit);                              % show the image
%     title(labels(i));                   % show the label
% end

N = 50; 
numCores = 1;
reachMethod = 'rstar-absdom-two';

[r0001, rb0001, cE0001, cands0001, vt0001] = nnv_net.evaluateRBN(RS_eps_norm_0001(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0001 = [0.001];
verify_time0001 = [sum(vt0001)];
safe0001 = [sum(rb0001==1)];
unsafe0001 = [sum(rb0001 == 0)];
unknown0001 = [sum(rb0001 == 2)];
T0001 = table(epsilon0001, safe0001, unsafe0001, unknown0001, verify_time0001)
fprintf('total time rstar norm (eps=0.001): %f ',verify_time0001);
save("verify_result/tanh_rstar_eps_0001_verify_norm.mat", 'T0001', 'r0001', 'rb0001', 'cE0001', 'cands0001', 'vt0001');

[r0002, rb0002, cE0002, cands0002, vt0002] = nnv_net.evaluateRBN(RS_eps_norm_0002(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0002 = [0.002];
verify_time0002 = [sum(vt0002)];
safe0002 = [sum(rb0002==1)];
unsafe0002 = [sum(rb0002 == 0)];
unknown0002 = [sum(rb0002 == 2)];
T0002 = table(epsilon0002, safe0002, unsafe0002, unknown0002, verify_time0002)
fprintf('total time rstar norm (eps=0.002): %f ',verify_time0002);
save("verify_result/tanh_rstar_eps_0002_verify_norm.mat", 'T0002', 'r0002', 'rb0002', 'cE0002', 'cands0002', 'vt0002');

[r0003, rb0003, cE0003, cands0003, vt0003] = nnv_net.evaluateRBN(RS_eps_norm_0003(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0003 = [0.003];
verify_time0003 = [sum(vt0003)];
safe0003 = [sum(rb0003==1)];
unsafe0003 = [sum(rb0003 == 0)];
unknown0003 = [sum(rb0003 == 2)];
T0003 = table(epsilon0003, safe0003, unsafe0003, unknown0003, verify_time0003)
fprintf('total time rstar norm (eps=0.003): %f ',verify_time0003);
save("verify_result/tanh_rstar_eps_0003_verify_norm.mat", 'T0003', 'r0003', 'rb0003', 'cE0003', 'cands0003', 'vt0003');

[r0004, rb0004, cE0004, cands0004, vt0004] = nnv_net.evaluateRBN(RS_eps_norm_0004(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0004 = [0.004];
verify_time0004 = [sum(vt0004)];
safe0004 = [sum(rb0004==1)];
unsafe0004 = [sum(rb0004 == 0)];
unknown0004 = [sum(rb0004 == 2)];
T0004 = table(epsilon0004, safe0004, unsafe0004, unknown0004, verify_time0004)
fprintf('total time rstar norm (eps=0.004): %f ',verify_time0004);
save("verify_result/tanh_rstar_eps_0004_verify_norm.mat", 'T0004', 'r0004', 'rb0004', 'cE0004', 'cands0004', 'vt0004');

[r0005, rb0005, cE0005, cands0005, vt0005] = nnv_net.evaluateRBN(RS_eps_norm_0005(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0005 = [0.005];
verify_time0005 = [sum(vt0005)];
safe0005 = [sum(rb0005==1)];
unsafe0005 = [sum(rb0005 == 0)];
unknown0005 = [sum(rb0005 == 2)];
T0005 = table(epsilon0005, safe0005, unsafe0005, unknown0005, verify_time0005)
fprintf('total time rstar norm (eps=0.005): %f ',verify_time0005);
save("verify_result/tanh_rstar_eps_0005_verify_norm.mat", 'T0005', 'r0005', 'rb0005', 'cE0005', 'cands0005', 'vt0005');

[r0006, rb0006, cE0006, cands0006, vt0006] = nnv_net.evaluateRBN(RS_eps_norm_0006(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0006 = [0.006];
verify_time0006 = [sum(vt0006)];
safe0006 = [sum(rb0006==1)];
unsafe0006 = [sum(rb0006 == 0)];
unknown0006 = [sum(rb0006 == 2)];
T0006 = table(epsilon0006, safe0006, unsafe0006, unknown0006, verify_time0006)
fprintf('total time rstar norm (eps=0.006): %f ',verify_time0006);
save("verify_result/tanh_rstar_eps_0006_verify_norm.mat", 'T0006', 'r0006', 'rb0006', 'cE0006', 'cands0006', 'vt0006');

[r0007, rb0007, cE0007, cands0007, vt0007] = nnv_net.evaluateRBN(RS_eps_norm_0007(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0007 = [0.007];
verify_time0007 = [sum(vt0007)];
safe0007 = [sum(rb0007==1)];
unsafe0007 = [sum(rb0007 == 0)];
unknown0007 = [sum(rb0007 == 2)];
T0007 = table(epsilon0007, safe0007, unsafe0007, unknown0007, verify_time0007)
fprintf('total time rstar norm (eps=0.007): %f ',verify_time0007);
save("verify_result/tanh_rstar_eps_0007_verify_norm.mat", 'T0007', 'r0007', 'rb0007', 'cE0007', 'cands0007', 'vt0007');

[r0008, rb0008, cE0008, cands0008, vt0008] = nnv_net.evaluateRBN(RS_eps_norm_0008(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0008 = [0.008];
verify_time0008 = [sum(vt0008)];
safe0008 = [sum(rb0008==1)];
unsafe0008 = [sum(rb0008 == 0)];
unknown0008 = [sum(rb0008 == 2)];
T0008 = table(epsilon0008, safe0008, unsafe0008, unknown0008, verify_time0008)
fprintf('total time rstar norm (eps=0.008): %f ',verify_time0008);
save("verify_result/tanh_rstar_eps_0008_verify_norm.mat", 'T0008', 'r0008', 'rb0008', 'cE0008', 'cands0008', 'vt0008');

[r0009, rb0009, cE0009, cands0009, vt0009] = nnv_net.evaluateRBN(RS_eps_norm_0009(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0009 = [0.009];
verify_time0009 = [sum(vt0009)];
safe0009 = [sum(rb0009==1)];
unsafe0009 = [sum(rb0009 == 0)];
unknown0009 = [sum(rb0009 == 2)];
T0009 = table(epsilon0009, safe0009, unsafe0009, unknown0009, verify_time0009)
fprintf('total time rstar norm (eps=0.009): %f ',verify_time0009);
save("verify_result/tanh_rstar_eps_0009_verify_norm.mat", 'T0009', 'r0009', 'rb0009', 'cE0009', 'cands0009', 'vt0009');

[r0010, rb0010, cE0010, cands0010, vt0010] = nnv_net.evaluateRBN(RS_eps_norm_0010(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0010 = [0.010];
verify_time0010 = [sum(vt0010)];
safe0010 = [sum(rb0010==1)];
unsafe0010 = [sum(rb0010 == 0)];
unknown0010 = [sum(rb0010 == 2)];
T0010 = table(epsilon0010, safe0010, unsafe0010, unknown0010, verify_time0010)
fprintf('total time rstar norm (eps=0.010): %f ',verify_time0010);
save("verify_result/tanh_rstar_eps_0010_verify_norm.mat", 'T0010', 'r0010', 'rb0010', 'cE0010', 'cands0010', 'vt0010');
