close all;
clear;
clc;

load not_normalized/inputStar005.mat S_eps_005
load not_normalized/inputStar010.mat S_eps_010 
load not_normalized/inputStar015.mat S_eps_015  
load not_normalized/inputStar020.mat S_eps_020 
load not_normalized/inputStar025.mat S_eps_025 
load not_normalized/inputStar030.mat S_eps_030

load not_normalized/inputStar1.mat S_eps_1
load not_normalized/inputStar2.mat S_eps_2
load not_normalized/inputStar3.mat S_eps_3
load not_normalized/inputStar4.mat S_eps_4
load not_normalized/inputStar5.mat S_eps_5
load not_normalized/inputStar6.mat S_eps_6
load not_normalized/inputStar7.mat S_eps_7
load not_normalized/inputStar8.mat S_eps_8
load not_normalized/inputStar9.mat S_eps_9
load not_normalized/inputStar10.mat S_eps_10


load MNIST_tanh_100_100_DenseNet.mat net
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[r1, rb1, cE1, cands1, vt1] = nnv_net.evaluateRBN(S_eps_1(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon1 = [0.1];
verify_time1 = [sum(vt1)];
safe1 = [sum(rb1==1)];
unsafe1 = [sum(rb1 == 0)];
unknown1 = [sum(rb1 == 2)];
T1 = table(epsilon1, safe1, unsafe1, unknown1, verify_time1)
fprintf('total time star (eps=0.1): %f ',verify_time1);
save("tanh_star_eps_1_verify.mat", 'T1', 'r1', 'rb1', 'cE1', 'cands1', 'vt1');

[r2, rb2, cE2, cands2, vt2] = nnv_net.evaluateRBN(S_eps_2(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon2 = [0.2];
verify_time2 = [sum(vt2)];
safe2 = [sum(rb2==1)];
unsafe2 = [sum(rb2 == 0)];
unknown2 = [sum(rb2 == 2)];
T2 = table(epsilon2, safe2, unsafe2, unknown2, verify_time2)
fprintf('total time star (eps=0.2): %f ',verify_time2);
save("tanh_star_eps_2_verify.mat", 'T2', 'r2', 'rb2', 'cE2', 'cands2', 'vt2');

[r3, rb3, cE3, cands3, vt3] = nnv_net.evaluateRBN(S_eps_3(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon3 = [0.3];
verify_time3 = [sum(vt3)];
safe3 = [sum(rb3==1)];
unsafe3 = [sum(rb3 == 0)];
unknown3 = [sum(rb3 == 2)];
T3 = table(epsilon3, safe3, unsafe3, unknown3, verify_time3)
fprintf('total time star (eps=0.3): %f ',verify_time3);
save("tanh_star_eps_3_verify.mat", 'T3', 'r3', 'rb3', 'cE3', 'cands3', 'vt3');

[r4, rb4, cE4, cands4, vt4] = nnv_net.evaluateRBN(S_eps_4(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon4 = [0.4];
verify_time4 = [sum(vt4)];
safe4 = [sum(rb4==1)];
unsafe4 = [sum(rb4 == 0)];
unknown4 = [sum(rb4 == 2)];
T4 = table(epsilon1, safe4, unsafe4, unknown4, verify_time4)
fprintf('total time star (eps=0.4): %f ',verify_time4);
save("tanh_star_eps_4_verify.mat", 'T4', 'r4', 'rb4', 'cE4', 'cands4', 'vt4');

[r5, rb5, cE5, cands5, vt5] = nnv_net.evaluateRBN(S_eps_5(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon5 = [0.5];
verify_time5 = [sum(vt5)];
safe5 = [sum(rb5==1)];
unsafe5 = [sum(rb5 == 0)];
unknown5 = [sum(rb5 == 2)];
T5 = table(epsilon5, safe5, unsafe5, unknown5, verify_time5)
fprintf('total time star (eps=0.5): %f ',verify_time5);
save("tanh_star_eps_5_verify.mat", 'T5', 'r5', 'rb5', 'cE5', 'cands5', 'vt5');

[r6, rb6, cE6, cands6, vt6] = nnv_net.evaluateRBN(S_eps_6(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon6 = [0.6];
verify_time6 = [sum(vt6)];
safe6 = [sum(rb6==1)];
unsafe6 = [sum(rb6 == 0)];
unknown6 = [sum(rb6 == 2)];
T6 = table(epsilon6, safe6, unsafe6, unknown6, verify_time6)
fprintf('total time star (eps=0.6): %f ',verify_time6);
save("tanh_star_eps_6_verify.mat", 'T6', 'r6', 'rb6', 'cE6', 'cands6', 'vt6');

[r7, rb7, cE7, cands7, vt7] = nnv_net.evaluateRBN(S_eps_7(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon7 = [0.7];
verify_time7 = [sum(vt7)];
safe7 = [sum(rb7==1)];
unsafe7 = [sum(rb7 == 0)];
unknown7 = [sum(rb7 == 2)];
T7 = table(epsilon7, safe7, unsafe7, unknown7, verify_time7)
fprintf('total time star (eps=0.7): %f ',verify_time7);
save("tanh_star_eps_7_verify.mat", 'T7', 'r7', 'rb7', 'cE7', 'cands7', 'vt7');

[r8, rb8, cE8, cands8, vt8] = nnv_net.evaluateRBN(S_eps_8(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon8 = [0.8];
verify_time8 = [sum(vt8)];
safe8 = [sum(rb8==1)];
unsafe8 = [sum(rb8 == 0)];
unknown8 = [sum(rb8 == 2)];
T8 = table(epsilon8, safe8, unsafe8, unknown8, verify_time8)
fprintf('total time star (eps=0.8): %f ',verify_time8);
save("tanh_star_eps_8_verify.mat", 'T8', 'r8', 'rb8', 'cE8', 'cands8', 'vt8');

[r9, rb9, cE9, cands9, vt9] = nnv_net.evaluateRBN(S_eps_9(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon9 = [0.9];
verify_time9 = [sum(vt9)];
safe9 = [sum(rb9==1)];
unsafe9 = [sum(rb9 == 0)];
unknown9 = [sum(rb9 == 2)];
T9 = table(epsilon9, safe9, unsafe9, unknown9, verify_time9)
fprintf('total time star (eps=0.9): %f ',verify_time9);
save("tanh_star_eps_9_verify.mat", 'T9', 'r9', 'rb9', 'cE9', 'cands9', 'vt9');

[r10, rb10, cE10, cands10, vt10] = nnv_net.evaluateRBN(S_eps_10(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon10 = [1.0];
verify_time10 = [sum(vt10)];
safe10 = [sum(rb10==1)];
unsafe10 = [sum(rb10 == 0)];
unknown10 = [sum(rb10 == 2)];
T10 = table(epsilon10, safe10, unsafe10, unknown10, verify_time10)
fprintf('total time star (eps=0.10): %f ',verify_time10);
save("tanh_star_eps_10_verify.mat", 'T10', 'r10', 'rb10', 'cE10', 'cands10', 'vt10');