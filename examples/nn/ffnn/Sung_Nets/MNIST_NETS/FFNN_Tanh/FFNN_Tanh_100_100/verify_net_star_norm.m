close all;
clear;
clc;

% load inputStar_norm005.mat;
% load inputStar_norm010.mat;
% load inputStar_norm015.mat;
% load inputStar_norm020.mat;
% load inputStar_norm025.mat;
% load inputStar_norm030.mat;

load tanh_100_100_normalized/inputStar_norm0001.mat; 
load tanh_100_100_normalized/inputStar_norm0002.mat;
load tanh_100_100_normalized/inputStar_norm0003.mat; 
load tanh_100_100_normalized/inputStar_norm0004.mat;
load tanh_100_100_normalized/inputStar_norm0005.mat; 
load tanh_100_100_normalized/inputStar_norm0006.mat;
load tanh_100_100_normalized/inputStar_norm0007.mat; 
load tanh_100_100_normalized/inputStar_norm0008.mat;
load tanh_100_100_normalized/inputStar_norm0009.mat; 
load tanh_100_100_normalized/inputStar_norm0010.mat; 

load MNIST_tanh_100_100_normalized_DenseNet.mat net;
L1 = LayerS(net.Layers(3).Weights, net.Layers(3).Bias, 'tansig');
L2 = LayerS(net.Layers(5).Weights, net.Layers(5).Bias, 'tansig');
L3 = LayerS(net.Layers(7).Weights, net.Layers(7).Bias, 'purelin');
nnv_net = FFNNS([L1 L2 L3]);

load tanh_100_100_images_normalized.mat;
labels = IM_labels;

figure                                          % initialize figure
colormap(gray)                                  % set to grayscale
for i = 1:50                                    % preview first 50 samples
    subplot(5,10,i);                              % plot them in 6 x 6 grid
    digit = reshape(S_eps_norm_0005(i).state_lb, [28,28]);     % row = 28 x 28 image
    imagesc(digit);                              % show the image
    title(labels(i));                   % show the label
end


N = 50; 
numCores = 1;
reachMethod = 'approx-star';

[r0001, rb0001, cE0001, cands0001, vt0001] = nnv_net.evaluateRBN(S_eps_norm_0001(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0001 = [0.001];
verify_time0001 = [sum(vt0001)];
safe0001 = [sum(rb0001==1)];
unsafe0001 = [sum(rb0001 == 0)];
unknown0001 = [sum(rb0001 == 2)];
T0001 = table(epsilon0001, safe0001, unsafe0001, unknown0001, verify_time0001)
fprintf('total time star norm (eps=0.001): %f ',verify_time0001);
save("verify_result/tanh_star_eps_0001_verify_norm.mat", 'T0001', 'r0001', 'rb0001', 'cE0001', 'cands0001', 'vt0001');

[r0002, rb0002, cE0002, cands0002, vt0002] = nnv_net.evaluateRBN(S_eps_norm_0002(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0002 = [0.002];
verify_time0002 = [sum(vt0002)];
safe0002 = [sum(rb0002==1)];
unsafe0002 = [sum(rb0002 == 0)];
unknown0002 = [sum(rb0002 == 2)];
T0002 = table(epsilon0002, safe0002, unsafe0002, unknown0002, verify_time0002)
fprintf('total time star norm (eps=0.002): %f ',verify_time0002);
save("verify_result/tanh_star_eps_0002_verify_norm.mat", 'T0002', 'r0002', 'rb0002', 'cE0002', 'cands0002', 'vt0002');

[r0003, rb0003, cE0003, cands0003, vt0003] = nnv_net.evaluateRBN(S_eps_norm_0003(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0003 = [0.003];
verify_time0003 = [sum(vt0003)];
safe0003 = [sum(rb0003==1)];
unsafe0003 = [sum(rb0003 == 0)];
unknown0003 = [sum(rb0003 == 2)];
T0003 = table(epsilon0003, safe0003, unsafe0003, unknown0003, verify_time0003)
fprintf('total time star norm (eps=0.003): %f ',verify_time0003);
save("verify_result/tanh_star_eps_0003_verify_norm.mat", 'T0003', 'r0003', 'rb0003', 'cE0003', 'cands0003', 'vt0003');

[r0004, rb0004, cE0004, cands0004, vt0004] = nnv_net.evaluateRBN(S_eps_norm_0004(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0004 = [0.004];
verify_time0004 = [sum(vt0004)];
safe0004 = [sum(rb0004==1)];
unsafe0004 = [sum(rb0004 == 0)];
unknown0004 = [sum(rb0004 == 2)];
T0004 = table(epsilon0004, safe0004, unsafe0004, unknown0004, verify_time0004)
fprintf('total time star norm (eps=0.004): %f ',verify_time0004);
save("verify_result/tanh_star_eps_0004_verify_norm.mat", 'T0004', 'r0004', 'rb0004', 'cE0004', 'cands0004', 'vt0004');

[r0005, rb0005, cE0005, cands0005, vt0005] = nnv_net.evaluateRBN(S_eps_norm_0005(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0005 = [0.005];
verify_time0005 = [sum(vt0005)];
safe0005 = [sum(rb0005==1)];
unsafe0005 = [sum(rb0005 == 0)];
unknown0005 = [sum(rb0005 == 2)];
T0005 = table(epsilon0005, safe0005, unsafe0005, unknown0005, verify_time0005)
fprintf('total time star norm (eps=0.005): %f ',verify_time0005);
save("verify_result/tanh_star_eps_0005_verify_norm.mat", 'T0005', 'r0005', 'rb0005', 'cE0005', 'cands0005', 'vt0005');

[r0006, rb0006, cE0006, cands0006, vt0006] = nnv_net.evaluateRBN(S_eps_norm_0006(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0006 = [0.006];
verify_time0006 = [sum(vt0006)];
safe0006 = [sum(rb0006==1)];
unsafe0006 = [sum(rb0006 == 0)];
unknown0006 = [sum(rb0006 == 2)];
T0006 = table(epsilon0006, safe0006, unsafe0006, unknown0006, verify_time0006)
fprintf('total time star norm (eps=0.006): %f ',verify_time0006);
save("verify_result/tanh_star_eps_0006_verify_norm.mat", 'T0006', 'r0006', 'rb0006', 'cE0006', 'cands0006', 'vt0006');

[r0007, rb0007, cE0007, cands0007, vt0007] = nnv_net.evaluateRBN(S_eps_norm_0007(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0007 = [0.007];
verify_time0007 = [sum(vt0007)];
safe0007 = [sum(rb0007==1)];
unsafe0007 = [sum(rb0007 == 0)];
unknown0007 = [sum(rb0007 == 2)];
T0007 = table(epsilon0007, safe0007, unsafe0007, unknown0007, verify_time0007)
fprintf('total time star norm (eps=0.007): %f ',verify_time0007);
save("verify_result/tanh_star_eps_0007_verify_norm.mat", 'T0007', 'r0007', 'rb0007', 'cE0007', 'cands0007', 'vt0007');

[r0008, rb0008, cE0008, cands0008, vt0008] = nnv_net.evaluateRBN(S_eps_norm_0008(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0008 = [0.008];
verify_time0008 = [sum(vt0008)];
safe0008 = [sum(rb0008==1)];
unsafe0008 = [sum(rb0008 == 0)];
unknown0008 = [sum(rb0008 == 2)];
T0008 = table(epsilon0008, safe0008, unsafe0008, unknown0008, verify_time0008)
fprintf('total time star norm (eps=0.008): %f ',verify_time0008);
save("verify_result/tanh_star_eps_0008_verify_norm.mat", 'T0008', 'r0008', 'rb0008', 'cE0008', 'cands0008', 'vt0008');

[r0009, rb0009, cE0009, cands0009, vt0009] = nnv_net.evaluateRBN(S_eps_norm_0009(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0009 = [0.009];
verify_time0009 = [sum(vt0009)];
safe0009 = [sum(rb0009==1)];
unsafe0009 = [sum(rb0009 == 0)];
unknown0009 = [sum(rb0009 == 2)];
T0009 = table(epsilon0009, safe0009, unsafe0009, unknown0009, verify_time0009)
fprintf('total time star norm (eps=0.009): %f ',verify_time0009);
save("verify_result/tanh_star_eps_0009_verify_norm.mat", 'T0009', 'r0009', 'rb0009', 'cE0009', 'cands0009', 'vt0009');

[r0010, rb0010, cE0010, cands0010, vt0010] = nnv_net.evaluateRBN(S_eps_norm_0010(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon0010 = [0.010];
verify_time0010 = [sum(vt0010)];
safe0010 = [sum(rb0010==1)];
unsafe0010 = [sum(rb0010 == 0)];
unknown0010 = [sum(rb0010 == 2)];
T0010 = table(epsilon0010, safe0010, unsafe0010, unknown0010, verify_time0010)
fprintf('total time star norm (eps=0.010): %f ',verify_time0010);
save("verify_result/tanh_star_eps_0010_verify_norm.mat", 'T0010', 'r0010', 'rb0010', 'cE0010', 'cands0010', 'vt0010');

% [r005, rb005, cE005, cands005, vt005] = nnv_net.evaluateRBN(S_eps_norm_005(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon005 = [0.005];
% verify_time005 = [sum(vt005)];
% safe005 = [sum(rb005==1)];
% unsafe005 = [sum(rb005 == 0)];
% unknown005 = [sum(rb005 == 2)];
% T005 = table(epsilon005, safe005, unsafe005, unknown005, verify_time005)
% fprintf('total time star norm (eps=0.005): %f ',verify_time005);
% save("tanh_star_eps_005_verify_norm.mat", 'T005', 'r005', 'rb005', 'cE005', 'cands005', 'vt005');
% 
% [r010, rb010, cE010, cands010, vt010] = nnv_net.evaluateRBN(S_eps_norm_010(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon010 = [0.010];
% verify_time010 = [sum(vt010)];
% safe010 = [sum(rb010==1)];
% unsafe010 = [sum(rb010 == 0)];
% unknown010 = [sum(rb010 == 2)];
% T010 = table(epsilon010, safe010, unsafe010, unknown010, verify_time010)
% fprintf('total time star norm (eps=0.010): %f ',verify_time010);
% save("tanh_star_eps_010_verify_norm.mat", 'T010', 'r010', 'rb010', 'cE010', 'cands010', 'vt010');
% 
% [r015, rb015, cE015, cands015, vt015] = nnv_net.evaluateRBN(S_eps_norm_015(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon015 = [0.015];
% verify_time015 = [sum(vt015)];
% safe015 = [sum(rb015==1)];
% unsafe015 = [sum(rb015 == 0)];
% unknown015 = [sum(rb015 == 2)];
% T015 = table(epsilon015, safe015, unsafe015, unknown015, verify_time015)
% fprintf('total time star norm (eps=0.015): %f ',verify_time015);
% save("tanh_star_eps_015_verify_norm.mat", 'T015', 'r015', 'rb015', 'cE015', 'cands015', 'vt015');
% 
% [r020, rb020, cE020, cands020, vt020] = nnv_net.evaluateRBN(S_eps_norm_020(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon020 = [0.020];
% verify_time020 = [sum(vt020)];
% safe020 = [sum(rb020==1)];
% unsafe020 = [sum(rb020 == 0)];
% unknown020 = [sum(rb020 == 2)];
% T020 = table(epsilon020, safe020, unsafe020, unknown020, verify_time020)
% fprintf('total time star norm (eps=0.020): %f ',verify_time020);
% save("tanh_star_eps_020_verify_norm.mat", 'T020', 'r020', 'rb020', 'cE020', 'cands020', 'vt020');
% 
% [r025, rb025, cE025, cands025, vt025] = nnv_net.evaluateRBN(S_eps_norm_025(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon025 = [0.025];
% verify_time025 = [sum(vt025)];
% safe025 = [sum(rb025==1)];
% unsafe025 = [sum(rb025 == 0)];
% unknown025 = [sum(rb025 == 2)];
% T025 = table(epsilon025, safe025, unsafe025, unknown025, verify_time025)
% fprintf('total time star norm (eps=0.025): %f ',verify_time025);
% save("tanh_star_eps_025_verify_norm.mat", 'T025', 'r025', 'rb025', 'cE025', 'cands025', 'vt025');
% 
% [r, rb, cE, cands, vt] = nnv_net.evaluateRBN(S_eps_norm_030(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon030 = [0.030];
% verify_time030 = [sum(vt030)];
% safe030 = [sum(rb030==1)];
% unsafe030 = [sum(rb030 == 0)];
% unknown030 = [sum(rb030 == 2)];
% T030 = table(epsilon030, safe030, unsafe030, unknown030, verify_time030)
% fprintf('total time star norm (eps=0.030): %f ',verify_time030);
% save("tanh_star_eps_030_verify_norm.mat", 'T030', 'r030', 'rb030', 'cE030', 'cands030', 'vt030');

