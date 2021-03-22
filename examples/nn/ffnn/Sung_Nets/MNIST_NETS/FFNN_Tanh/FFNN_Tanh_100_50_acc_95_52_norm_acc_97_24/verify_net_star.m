close all;
clear;
clc;

load not_normalized/inputStar005.mat S_eps_005
load not_normalized/inputStar010.mat S_eps_010 
load not_normalized/inputStar015.mat S_eps_015  
load not_normalized/inputStar020.mat S_eps_020 
load not_normalized/inputStar025.mat S_eps_025 
load not_normalized/inputStar030.mat S_eps_030

load MNIST_tanh_100_50_DenseNet.mat net
L1 = LayerS(net.Layers(3).Weights, net.Layers(3).Bias, 'tansig');
L2 = LayerS(net.Layers(5).Weights, net.Layers(5).Bias, 'tansig');
L3 = LayerS(net.Layers(7).Weights, net.Layers(7).Bias, 'purelin');
nnv_net = FFNNS([L1 L2 L3]);

load images.mat;
labels = IM_labels;

figure                                          % initialize figure
colormap(gray)                                  % set to grayscale
for i = 1:50                                    % preview first 50 samples
    subplot(5,10,i);                              % plot them in 6 x 6 grid
    digit = reshape(S_eps_005(i).state_lb, [28,28]);     % row = 28 x 28 image
    imagesc(digit);                              % show the image
    title(labels(i));                   % show the label
end

N = 50; 
numCores = 1;
reachMethod = 'approx-star';

[r005, rb005, cE005, cands005, vt005] = nnv_net.evaluateRBN(S_eps_005(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon005 = [0.005];
verify_time005 = [sum(vt005)];
safe005 = [sum(rb005==1)];
unsafe005 = [sum(rb005 == 0)];
unknown005 = [sum(rb005 == 2)];
T005 = table(epsilon005, safe005, unsafe005, unknown005, verify_time005)
fprintf('total time star (eps=0.005): %f ',verify_time005);
save("tanh_star_eps_005_verify.mat", 'T005', 'r005', 'rb005', 'cE005', 'cands005', 'vt005');

[r010, rb010, cE010, cands010, vt010] = nnv_net.evaluateRBN(S_eps_010(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon010 = [0.010];
verify_time010 = [sum(vt010)];
safe010 = [sum(rb010==1)];
unsafe010 = [sum(rb010 == 0)];
unknown010 = [sum(rb010 == 2)];
T010 = table(epsilon010, safe010, unsafe010, unknown010, verify_time010)
fprintf('total time star (eps=0.010): %f ',verify_time010);
save("tanh_star_eps_010_verify.mat", 'T010', 'r010', 'rb010', 'cE010', 'cands010', 'vt010');

[r015, rb015, cE015, cands015, vt015] = nnv_net.evaluateRBN(S_eps_015(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon015 = [0.015];
verify_time015 = [sum(vt015)];
safe015 = [sum(rb015==1)];
unsafe015 = [sum(rb015 == 0)];
unknown015 = [sum(rb015 == 2)];
T015 = table(epsilon015, safe015, unsafe015, unknown015, verify_time015)
fprintf('total time star (eps=0.015): %f ',verify_time015);
save("tanh_star_eps_015_verify.mat", 'T015', 'r015', 'rb015', 'cE015', 'cands015', 'vt015');

[r020, rb020, cE020, cands020, vt020] = nnv_net.evaluateRBN(S_eps_020(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon020 = [0.020];
verify_time020 = [sum(vt020)];
safe020 = [sum(rb020==1)];
unsafe020 = [sum(rb020 == 0)];
unknown020 = [sum(rb020 == 2)];
T020 = table(epsilon020, safe020, unsafe020, unknown020, verify_time020)
fprintf('total time star (eps=0.020): %f ',verify_time020);
save("tanh_star_eps_020_verify.mat", 'T020', 'r020', 'rb020', 'cE020', 'cands020', 'vt020');

[r025, rb025, cE025, cands025, vt025] = nnv_net.evaluateRBN(S_eps_025(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon025 = [0.025];
verify_time025 = [sum(vt025)];
safe025 = [sum(rb025==1)];
unsafe025 = [sum(rb025 == 0)];
unknown025 = [sum(rb025 == 2)];
T025 = table(epsilon025, safe025, unsafe025, unknown025, verify_time025)
fprintf('total time star (eps=0.025): %f ',verify_time025);
save("tanh_star_eps_025_verify.mat", 'T025', 'r025', 'rb025', 'cE025', 'cands025', 'vt025');

[r030, rb030, cE030, cands030, vt030] = nnv_net.evaluateRBN(S_eps_030(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon030 = [0.030];
verify_time030 = [sum(vt030)];
safe030 = [sum(rb030==1)];
unsafe030 = [sum(rb030 == 0)];
unknown030 = [sum(rb030 == 2)];
T030 = table(epsilon030, safe030, unsafe030, unknown030, verify_time030)
fprintf('total time star (eps=0.030): %f ',verify_time030);
save("tanh_star_eps_030_verify.mat", 'T030', 'r030', 'rb030', 'cE030', 'cands030', 'vt030');

