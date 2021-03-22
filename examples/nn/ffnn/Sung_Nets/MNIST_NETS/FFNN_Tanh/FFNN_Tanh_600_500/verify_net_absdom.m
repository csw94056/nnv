close all;
clear;
clc;

load not_normalized/inputAbsDom005.mat A_eps_005 
load not_normalized/inputAbsDom010.mat A_eps_010 
load not_normalized/inputAbsDom015.mat A_eps_015 
load not_normalized/inputAbsDom020.mat A_eps_020 
load not_normalized/inputAbsDom025.mat A_eps_025 
load not_normalized/inputAbsDom030.mat A_eps_030
load not_normalized/inputAbsDom1.mat A_eps_1 
load not_normalized/inputAbsDom2.mat A_eps_2 
load not_normalized/inputAbsDom3.mat A_eps_3 
load not_normalized/inputAbsDom4.mat A_eps_4 
load not_normalized/inputAbsDom5.mat A_eps_5 

load MNIST_tanh_600_500_DenseNet.mat net;
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
    digit = reshape(A_eps_005(i).lb{1}, [28,28]);     % row = 28 x 28 image
    imagesc(digit);                              % show the image
    title(labels(i));                   % show the label
end

N = 50; 
numCores = 1;
reachMethod = 'absdom';

[r005, rb005, cE005, cands005, vt005] = nnv_net.evaluateRBN(A_eps_005(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon005 = [0.005];
verify_time005 = [sum(vt005)];
safe005 = [sum(rb005==1)];
unsafe005 = [sum(rb005 == 0)];
unknown005 = [sum(rb005 == 2)];
T005 = table(epsilon005, safe005, unsafe005, unknown005, verify_time005)
fprintf('total time absdom (eps=0.005): %f ',verify_time005);
save("tanh_absdom_eps_005_verify.mat", 'T005', 'r005', 'rb005', 'cE005', 'cands005', 'vt005');

[r010, rb010, cE010, cands010, vt010] = nnv_net.evaluateRBN(A_eps_010(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon010 = [0.010];
verify_time010 = [sum(vt010)];
safe010 = [sum(rb010==1)];
unsafe010 = [sum(rb010 == 0)];
unknown010 = [sum(rb010 == 2)];
T010 = table(epsilon010, safe010, unsafe010, unknown010, verify_time010)
fprintf('total time absdom (eps=0.010): %f ',verify_time010);
save("tanh_absdom_eps_010_verify.mat", 'T010', 'r010', 'rb010', 'cE010', 'cands010', 'vt010');

[r015, rb015, cE015, cands015, vt015] = nnv_net.evaluateRBN(A_eps_015(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon015 = [0.015];
verify_time015 = [sum(vt015)];
safe015 = [sum(rb015==1)];
unsafe015 = [sum(rb015 == 0)];
unknown015 = [sum(rb015 == 2)];
T015 = table(epsilon015, safe015, unsafe015, unknown015, verify_time015)
fprintf('total time absdom (eps=0.015): %f ',verify_time015);
save("tanh_absdom_eps_015_verify.mat", 'T015', 'r015', 'rb015', 'cE015', 'cands015', 'vt015');

[r020, rb020, cE020, cands020, vt020] = nnv_net.evaluateRBN(A_eps_020(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon020 = [0.020];
verify_time020 = [sum(vt020)];
safe020 = [sum(rb020==1)];
unsafe020 = [sum(rb020 == 0)];
unknown020 = [sum(rb020 == 2)];
T020 = table(epsilon020, safe020, unsafe020, unknown020, verify_time020)
fprintf('total time absdom (eps=0.020): %f ',verify_time020);
save("tanh_absdom_eps_020_verify.mat", 'T020', 'r020', 'rb020', 'cE020', 'cands020', 'vt020');

[r025, rb025, cE025, cands025, vt025] = nnv_net.evaluateRBN(A_eps_025(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon025 = [0.025];
verify_time025 = [sum(vt025)];
safe025 = [sum(rb025==1)];
unsafe025 = [sum(rb025 == 0)];
unknown025 = [sum(rb025 == 2)];
T025 = table(epsilon025, safe025, unsafe025, unknown025, verify_time025)
fprintf('total time absdom (eps=0.025): %f ',verify_time025);
save("tanh_absdom_eps_025_verify.mat", 'T025', 'r025', 'rb025', 'cE025', 'cands025', 'vt025');

[r030, rb030, cE030, cands030, vt030] = nnv_net.evaluateRBN(A_eps_030(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon030 = [0.030];
verify_time030 = [sum(vt030)];
safe030 = [sum(rb030==1)];
unsafe030 = [sum(rb030 == 0)];
unknown030 = [sum(rb030 == 2)];
T030 = table(epsilon030, safe030, unsafe030, unknown030, verify_time030)
fprintf('total time absdom (eps=0.030): %f ',verify_time030);
save("tanh_absdom_eps_030_verify.mat", 'T030', 'r030', 'rb030', 'cE030', 'cands030', 'vt030');

[r1, rb1, cE1, cands1, vt1] = nnv_net.evaluateRBN(A_eps_1(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon1 = [1];
verify_time1 = [sum(vt1)];
safe1 = [sum(rb1==1)];
unsafe1 = [sum(rb1 == 0)];
unknown1 = [sum(rb1 == 2)];
T1 = table(epsilon1, safe1, unsafe1, unknown1, verify_time1)
fprintf('total time absdom (eps=1): %f ',verify_time1);
save("tanh_absdom_eps_1_verify.mat", 'T1', 'r1', 'rb1', 'cE1', 'cands1', 'vt1');

[r2, rb2, cE2, cands2, vt2] = nnv_net.evaluateRBN(A_eps_2(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon2 = [1];
verify_time2 = [sum(vt2)];
safe2 = [sum(rb2==1)];
unsafe2 = [sum(rb2 == 0)];
unknown2 = [sum(rb2 == 2)];
T2 = table(epsilon2, safe2, unsafe2, unknown2, verify_time2)
fprintf('total time absdom (eps=2): %f ',verify_time2);
save("tanh_absdom_eps_2_verify.mat", 'T2', 'r2', 'rb2', 'cE2', 'cands2', 'vt2');

[r3, rb3, cE3, cands3, vt3] = nnv_net.evaluateRBN(A_eps_3(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon3 = [3];
verify_time3 = [sum(vt3)];
safe3 = [sum(rb3==1)];
unsafe3 = [sum(rb3 == 0)];
unknown3 = [sum(rb3 == 2)];
T3 = table(epsilon3, safe3, unsafe3, unknown3, verify_time3)
fprintf('total time absdom (eps=3): %f ',verify_time3);
save("tanh_absdom_eps_3_verify.mat", 'T3', 'r3', 'rb3', 'cE3', 'cands3', 'vt3');

[r4, rb4, cE4, cands4, vt4] = nnv_net.evaluateRBN(A_eps_4(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon4 = [4];
verify_time4 = [sum(vt4)];
safe4 = [sum(rb4==1)];
unsafe4 = [sum(rb4 == 0)];
unknown4 = [sum(rb4 == 2)];
T4 = table(epsilon4, safe4, unsafe4, unknown4, verify_time4)
fprintf('total time absdom (eps=4): %f ',verify_time4);
save("tanh_absdom_eps_4_verify.mat", 'T4', 'r4', 'rb4', 'cE4', 'cands4', 'vt4');

[r5, rb5, cE5, cands5, vt5] = nnv_net.evaluateRBN(A_eps_5(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon5 = [5];
verify_time5 = [sum(vt5)];
safe5 = [sum(rb5==1)];
unsafe5 = [sum(rb5 == 0)];
unknown5 = [sum(rb5 == 2)];
T5 = table(epsilon5, safe5, unsafe5, unknown5, verify_time5)
fprintf('total time absdom (eps=5): %f ',verify_time5);
save("tanh_absdom_eps_5_verify.mat", 'T5', 'r5', 'rb5', 'cE5', 'cands5', 'vt5');


