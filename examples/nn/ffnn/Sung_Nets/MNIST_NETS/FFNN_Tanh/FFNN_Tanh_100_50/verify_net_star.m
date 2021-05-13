close all;
clear;
clc;

load tanh_100_50_not_normalized/inputStar02.mat;
load tanh_100_50_not_normalized/inputStar04.mat;
load tanh_100_50_not_normalized/inputStar06.mat;
load tanh_100_50_not_normalized/inputStar08.mat;
load tanh_100_50_not_normalized/inputStar10.mat;
load tanh_100_50_not_normalized/inputStar12.mat;
load tanh_100_50_not_normalized/inputStar14.mat;
load tanh_100_50_not_normalized/inputStar16.mat;
load tanh_100_50_not_normalized/inputStar18.mat;
load tanh_100_50_not_normalized/inputStar20.mat;
load tanh_100_50_not_normalized/inputStar22.mat;
load tanh_100_50_not_normalized/inputStar24.mat;
load tanh_100_50_not_normalized/inputStar26.mat;

load MNIST_tanh_100_50_DenseNet.mat net
L1 = LayerS(net.Layers(3).Weights, net.Layers(3).Bias, 'tansig');
L2 = LayerS(net.Layers(5).Weights, net.Layers(5).Bias, 'tansig');
L3 = LayerS(net.Layers(7).Weights, net.Layers(7).Bias, 'purelin');
nnv_net = FFNNS([L1 L2 L3]);

load tanh_100_50_images.mat;
labels = IM_labels;

% figure                                          % initialize figure
% colormap(gray)                                  % set txxxo grayscale
% for i = 1:50                                    % preview first 50 samples
%     subplot(5,10,i);                              % plot them in 6 x 6 grid
%     digit = reshape(S_eps_005(i).state_lb, [28,28]);     % row = 28 x 28 image
%     imagesc(digit);                              % show the image
%     title(labels(i));                   % show the label
% end

N = [1,3,4,5,6,7,8,10,11,12,13,14,15,16,17,19,21,22,23,24,25,26,...
    27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,52,53,54,55,56];
numCores = 1;
reachMethod = 'approx-star';

[r02, rb02, cE02, cands02, vt02] = nnv_net.evaluateRBN(S_eps_02(N), labels(N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon02 = [0.2];
verify_time02 = [sum(vt02)];
safe02 = [sum(rb02==1)];
unsafe02 = [sum(rb02 == 0)];
unknown02 = [sum(rb02 == 2)];
T02 = table(epsilon02, safe02, unsafe02, unknown02, verify_time02)
fprintf('total time star (eps=0.2): %f ',verify_time02);
save("verify_result/tanh_star_eps_02_verify.mat", 'T02', 'r02', 'rb02', 'cE02', 'cands02', 'vt02');


[r04, rb04, cE04, cands04, vt04] = nnv_net.evaluateRBN(S_eps_04(N), labels(N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon04 = [0.4];
verify_time04 = [sum(vt04)];
safe04 = [sum(rb04==1)];
unsafe04 = [sum(rb04 == 0)];
unknown04 = [sum(rb04 == 2)];
T04 = table(epsilon04, safe04, unsafe04, unknown04, verify_time04)
fprintf('total time star (eps=0.4): %f ',verify_time04);
save("verify_result/tanh_star_eps_04_verify.mat", 'T04', 'r04', 'rb04', 'cE04', 'cands04', 'vt04');


[r06, rb06, cE06, cands06, vt06] = nnv_net.evaluateRBN(S_eps_06(N), labels(N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon06 = [0.6];
verify_time06 = [sum(vt06)];
safe06 = [sum(rb06==1)];
unsafe06 = [sum(rb06 == 0)];
unknown06 = [sum(rb06 == 2)];
T06 = table(epsilon06, safe06, unsafe06, unknown06, verify_time06)
fprintf('total time star (eps=0.6): %f ',verify_time06);
save("verify_result/tanh_star_eps_06_verify.mat", 'T06', 'r06', 'rb06', 'cE06', 'cands06', 'vt06');

[r08, rb08, cE08, cands08, vt08] = nnv_net.evaluateRBN(S_eps_08(N), labels(N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon08 = [0.8];
verify_time08 = [sum(vt08)];
safe08 = [sum(rb08==1)];
unsafe08 = [sum(rb08 == 0)];
unknown08 = [sum(rb08 == 2)];
T08 = table(epsilon08, safe08, unsafe08, unknown08, verify_time08)
fprintf('total time star (eps=0.8): %f ',verify_time08);
save("verify_result/tanh_star_eps_08_verify.mat", 'T08', 'r08', 'rb08', 'cE08', 'cands08', 'vt08');

[r10, rb10, cE10, cands10, vt10] = nnv_net.evaluateRBN(S_eps_10(N), labels(N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon10 = [1.0];
verify_time10 = [sum(vt10)];
safe10 = [sum(rb10==1)];
unsafe10 = [sum(rb10 == 0)];
unknown10 = [sum(rb10 == 2)];
T10 = table(epsilon10, safe10, unsafe10, unknown10, verify_time10)
fprintf('total time star (eps=1.0): %f ',verify_time10);
save("verify_result/tanh_star_eps_10_verify.mat", 'T10', 'r10', 'rb10', 'cE10', 'cands10', 'vt10');


[r12, rb12, cE12, cands12, vt12] = nnv_net.evaluateRBN(S_eps_12(N), labels(N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon12 = [1.2];
verify_time12 = [sum(vt12)];
safe12 = [sum(rb12==1)];
unsafe12 = [sum(rb12 == 0)];
unknown12 = [sum(rb12 == 2)];
T12 = table(epsilon12, safe12, unsafe12, unknown12, verify_time12)
fprintf('total time star (eps=1.2): %f ',verify_time12);
save("verify_result/tanh_star_eps_12_verify.mat", 'T12', 'r12', 'rb12', 'cE12', 'cands12', 'vt12');


[r14, rb14, cE14, cands14, vt14] = nnv_net.evaluateRBN(S_eps_14(N), labels(N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon14 = [1.4];
verify_time14 = [sum(vt14)];
safe14 = [sum(rb14==1)];
unsafe14 = [sum(rb14 == 0)];
unknown14 = [sum(rb14 == 2)];
T14 = table(epsilon14, safe14, unsafe14, unknown14, verify_time14)
fprintf('total time star (eps=1.4): %f ',verify_time14);
save("verify_result/tanh_star_eps_14_verify.mat", 'T14', 'r14', 'rb14', 'cE14', 'cands14', 'vt14');

[r16, rb16, cE16, cands16, vt16] = nnv_net.evaluateRBN(S_eps_16(N), labels(N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon16 = [1.6];
verify_time16 = [sum(vt16)];
safe16 = [sum(rb16==1)];
unsafe16 = [sum(rb16 == 0)];
unknown16 = [sum(rb16 == 2)];
T16 = table(epsilon16, safe16, unsafe16, unknown16, verify_time16)
fprintf('total time star (eps=1.6): %f ',verify_time16);
save("verify_result/tanh_star_eps_16_verify.mat", 'T16', 'r16', 'rb16', 'cE16', 'cands16', 'vt16');

[r18, rb18, cE18, cands18, vt18] = nnv_net.evaluateRBN(S_eps_18(N), labels(N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon18 = [1.8];
verify_time18 = [sum(vt18)];
safe18 = [sum(rb18==1)];
unsafe18 = [sum(rb18 == 0)];
unknown18 = [sum(rb18 == 2)];
T18 = table(epsilon18, safe18, unsafe18, unknown18, verify_time18)
fprintf('total time star (eps=1.8): %f ',verify_time18);
save("verify_result/tanh_star_eps_18_verify.mat", 'T18', 'r18', 'rb18', 'cE18', 'cands18', 'vt18');

[r20, rb20, cE20, cands20, vt20] = nnv_net.evaluateRBN(S_eps_20(N), labels(N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon20 = [2.0];
verify_time20 = [sum(vt20)];
safe20 = [sum(rb20==1)];
unsafe20 = [sum(rb20 == 0)];
unknown20 = [sum(rb20 == 2)];
T20 = table(epsilon20, safe20, unsafe20, unknown20, verify_time20)
fprintf('total time star (eps=2.0): %f ',verify_time20);
save("verify_result/tanh_star_eps_20_verify.mat", 'T20', 'r20', 'rb20', 'cE20', 'cands20', 'vt20');

[r22, rb22, cE22, cands22, vt22] = nnv_net.evaluateRBN(S_eps_22(N), labels(N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon22 = [2.2];
verify_time22 = [sum(vt22)];
safe22 = [sum(rb22==1)];
unsafe22 = [sum(rb22 == 0)];
unknown22 = [sum(rb22 == 2)];
T22 = table(epsilon22, safe22, unsafe22, unknown22, verify_time22)
fprintf('total time star (eps=2.2): %f ',verify_time22);
save("verify_result/tanh_star_eps_22_verify.mat", 'T22', 'r22', 'rb22', 'cE22', 'cands22', 'vt22');

[r24, rb24, cE24, cands24, vt24] = nnv_net.evaluateRBN(S_eps_24(N), labels(N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon24 = [2.4];
verify_time24 = [sum(vt24)];
safe24 = [sum(rb24==1)];
unsafe24 = [sum(rb24 == 0)];
unknown24 = [sum(rb24 == 2)];
T24 = table(epsilon24, safe24, unsafe24, unknown24, verify_time24)
fprintf('total time star (eps=2.4): %f ',verify_time24);
save("verify_result/tanh_star_eps_24_verify.mat", 'T24', 'r24', 'rb24', 'cE24', 'cands24', 'vt24');

[r26, rb26, cE26, cands26, vt26] = nnv_net.evaluateRBN(S_eps_26(N), labels(N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon26 = [2.6];
verify_time26 = [sum(vt26)];
safe26 = [sum(rb26==1)];
unsafe26 = [sum(rb26 == 0)];
unknown26 = [sum(rb26 == 2)];
T26 = table(epsilon26, safe26, unsafe26, unknown26, verify_time26)
fprintf('total time star (eps=2.6): %f ',verify_time26);
save("verify_result/tanh_star_eps_26_verify.mat", 'T26', 'r26', 'rb26', 'cE26', 'cands26', 'vt26');