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
load not_normalized/inputAbsDom6.mat A_eps_6
load not_normalized/inputAbsDom7.mat A_eps_7
load not_normalized/inputAbsDom8.mat A_eps_8
load not_normalized/inputAbsDom9.mat A_eps_9
load not_normalized/inputAbsDom10.mat A_eps_10


load MNIST_tanh_100_50_DenseNet.mat net;
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

% [r001, rb001, cE001, cands001, vt001] = nnv_net.evaluateRBN(A_eps_001(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon001 = [0.001];
% verify_time001 = [sum(vt001)];
% safe001 = [sum(rb001==1)];
% unsafe001 = [sum(rb001 == 0)];
% unknown001 = [sum(rb001 == 2)];
% T001 = table(epsilon001, safe001, unsafe001, unknown001, verify_time001)
% fprintf('total time absdom (eps=0.001): %f ',verify_time001);
% save("tanh_absdom_eps_001_verify.mat", 'T001', 'r001', 'rb001', 'cE001', 'cands001', 'vt001');
% 
% 
% [r002, rb002, cE002, cands002, vt002] = nnv_net.evaluateRBN(A_eps_002(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon002 = [0.002];
% verify_time002 = [sum(vt002)];
% safe002 = [sum(rb002==1)];
% unsafe002 = [sum(rb002 == 0)];
% unknown002 = [sum(rb002 == 2)];
% T002 = table(epsilon002, safe002, unsafe002, unknown002, verify_time002)
% fprintf('total time absdom (eps=0.002): %f ',verify_time002);
% save("tanh_absdom_eps_002_verify.mat", 'T002', 'r002', 'rb002', 'cE002', 'cands002', 'vt002');
% 
% 
% [r003, rb003, cE003, cands003, vt003] = nnv_net.evaluateRBN(A_eps_003(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon003 = [0.003];
% verify_time003 = [sum(vt003)];
% safe003 = [sum(rb003==1)];
% unsafe003 = [sum(rb003 == 0)];
% unknown003 = [sum(rb003 == 2)];
% T003 = table(epsilon003, safe003, unsafe003, unknown003, verify_time003)
% fprintf('total time absdom (eps=0.003): %f ',verify_time003);
% save("tanh_absdom_eps_003_verify.mat", 'T003', 'r003', 'rb003', 'cE003', 'cands003', 'vt003');
% 
% 
% [r004, rb004, cE004, cands004, vt004] = nnv_net.evaluateRBN(A_eps_004(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon004 = [0.004];
% verify_time004 = [sum(vt004)];
% safe004 = [sum(rb004==1)];
% unsafe004 = [sum(rb004 == 0)];
% unknown004 = [sum(rb004 == 2)];
% T004 = table(epsilon004, safe004, unsafe004, unknown004, verify_time004)
% fprintf('total time absdom (eps=0.004): %f ',verify_time004);
% save("tanh_absdom_eps_004_verify.mat", 'T004', 'r004', 'rb004', 'cE004', 'cands004', 'vt004');

[r005, rb005, cE005, cands005, vt005] = nnv_net.evaluateRBN(A_eps_005(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon005 = [0.005];
verify_time005 = [sum(vt005)];
safe005 = [sum(rb005==1)];
unsafe005 = [sum(rb005 == 0)];
unknown005 = [sum(rb005 == 2)];
T005 = table(epsilon005, safe005, unsafe005, unknown005, verify_time005)
fprintf('total time absdom (eps=0.005): %f ',verify_time005);
save("tanh_absdom_eps_005_verify.mat", 'T005', 'r005', 'rb005', 'cE005', 'cands005', 'vt005');


% [r006, rb006, cE006, cands006, vt006] = nnv_net.evaluateRBN(A_eps_006(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon006 = [0.006];
% verify_time006 = [sum(vt006)];
% safe006 = [sum(rb006==1)];
% unsafe006 = [sum(rb006 == 0)];
% unknown006 = [sum(rb006 == 2)];
% T006 = table(epsilon006, safe006, unsafe006, unknown006, verify_time006)
% fprintf('total time absdom (eps=0.006): %f ',verify_time006);
% save("tanh_absdom_eps_006_verify.mat", 'T006', 'r006', 'rb006', 'cE006', 'cands006', 'vt006');
% 
% 
% [r007, rb007, cE007, cands007, vt007] = nnv_net.evaluateRBN(A_eps_007(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon007 = [0.007];
% verify_time007 = [sum(vt007)];
% safe007 = [sum(rb007 ==1)];
% unsafe007 = [sum(rb007 == 0)];
% unknown007 = [sum(rb007 == 2)];
% T007 = table(epsilon007, safe007, unsafe007, unknown007, verify_time007)
% fprintf('total time absdom (eps=0.007): %f ',verify_time007);
% save("tanh_absdom_eps_007_verify.mat", 'T007', 'r007', 'rb007', 'cE007', 'cands007', 'vt007');
% 
% 
% [r008, rb008, cE008, cands008, vt008] = nnv_net.evaluateRBN(A_eps_008(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon008 = [0.008];
% verify_time008 = [sum(vt008)];
% safe008 = [sum(rb008==1)];
% unsafe008 = [sum(rb008 == 0)];
% unknown008 = [sum(rb008 == 2)];
% T008 = table(epsilon008, safe008, unsafe008, unknown008, verify_time008)
% fprintf('total time absdom (eps=0.008): %f ',verify_time008);
% save("tanh_absdom_eps_008_verify.mat", 'T008', 'r008', 'rb008', 'cE008', 'cands008', 'vt008');
% 
% 
% [r009, rb009, cE009, cands009, vt009] = nnv_net.evaluateRBN(A_eps_009(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon009 = [0.009];
% verify_time009 = [sum(vt009)];
% safe009 = [sum(rb009 == 1)];
% unsafe009 = [sum(rb009 == 0)];
% unknown009 = [sum(rb009 == 2)];
% T009 = table(epsilon009, safe009, unsafe009, unknown009, verify_time009)
% fprintf('total time absdom (eps=0.009): %f ',verify_time009);
% save("tanh_absdom_eps_009_verify.mat", 'T009', 'r009', 'rb009', 'cE009', 'cands009', 'vt009');

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[r1, rb1, cE1, cands1, vt1] = nnv_net.evaluateRBN(A_eps_1(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon1 = [0.1];
verify_time1 = [sum(vt1)];
safe1 = [sum(rb1==1)];
unsafe1 = [sum(rb1 == 0)];
unknown1 = [sum(rb1 == 2)];
T1 = table(epsilon1, safe1, unsafe1, unknown1, verify_time1)
fprintf('total time absdom (eps=0.1): %f ',verify_time1);
save("tanh_absdom_eps_1_verify.mat", 'T1', 'r1', 'rb1', 'cE1', 'cands1', 'vt1');

[r2, rb2, cE2, cands2, vt2] = nnv_net.evaluateRBN(A_eps_2(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon2 = [0.2];
verify_time2 = [sum(vt2)];
safe2 = [sum(rb2==1)];
unsafe2 = [sum(rb2 == 0)];
unknown2 = [sum(rb2 == 2)];
T2 = table(epsilon2, safe2, unsafe2, unknown2, verify_time2)
fprintf('total time absdom (eps=0.2): %f ',verify_time2);
save("tanh_absdom_eps_2_verify.mat", 'T2', 'r2', 'rb2', 'cE2', 'cands2', 'vt2');

[r3, rb3, cE3, cands3, vt3] = nnv_net.evaluateRBN(A_eps_3(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon3 = [0.3];
verify_time3 = [sum(vt3)];
safe3 = [sum(rb3==1)];
unsafe3 = [sum(rb3 == 0)];
unknown3 = [sum(rb3 == 2)];
T3 = table(epsilon3, safe3, unsafe3, unknown3, verify_time3)
fprintf('total time absdom (eps=0.3): %f ',verify_time3);
save("tanh_absdom_eps_3_verify.mat", 'T3', 'r3', 'rb3', 'cE3', 'cands3', 'vt3');

[r4, rb4, cE4, cands4, vt4] = nnv_net.evaluateRBN(A_eps_4(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon4 = [0.4];
verify_time4 = [sum(vt4)];
safe4 = [sum(rb4==1)];
unsafe4 = [sum(rb4 == 0)];
unknown4 = [sum(rb4 == 2)];
T4 = table(epsilon1, safe4, unsafe4, unknown4, verify_time4)
fprintf('total time absdom (eps=0.4): %f ',verify_time4);
save("tanh_absdom_eps_4_verify.mat", 'T4', 'r4', 'rb4', 'cE4', 'cands4', 'vt4');

[r5, rb5, cE5, cands5, vt5] = nnv_net.evaluateRBN(A_eps_5(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon5 = [0.5];
verify_time5 = [sum(vt5)];
safe5 = [sum(rb5==1)];
unsafe5 = [sum(rb5 == 0)];
unknown5 = [sum(rb5 == 2)];
T5 = table(epsilon5, safe5, unsafe5, unknown5, verify_time5)
fprintf('total time absdom (eps=0.5): %f ',verify_time5);
save("tanh_absdom_eps_5_verify.mat", 'T5', 'r5', 'rb5', 'cE5', 'cands5', 'vt5');

[r6, rb6, cE6, cands6, vt6] = nnv_net.evaluateRBN(A_eps_6(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon6 = [0.6];
verify_time6 = [sum(vt6)];
safe6 = [sum(rb6==1)];
unsafe6 = [sum(rb6 == 0)];
unknown6 = [sum(rb6 == 2)];
T6 = table(epsilon6, safe6, unsafe6, unknown6, verify_time6)
fprintf('total time absdom (eps=0.6): %f ',verify_time6);
save("tanh_absdom_eps_6_verify.mat", 'T6', 'r6', 'rb6', 'cE6', 'cands6', 'vt6');

[r7, rb7, cE7, cands7, vt7] = nnv_net.evaluateRBN(A_eps_7(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon7 = [0.7];
verify_time7 = [sum(vt7)];
safe7 = [sum(rb7==1)];
unsafe7 = [sum(rb7 == 0)];
unknown7 = [sum(rb7 == 2)];
T7 = table(epsilon7, safe7, unsafe7, unknown7, verify_time7)
fprintf('total time absdom (eps=0.7): %f ',verify_time7);
save("tanh_absdom_eps_7_verify.mat", 'T7', 'r7', 'rb7', 'cE7', 'cands7', 'vt7');

[r8, rb8, cE8, cands8, vt8] = nnv_net.evaluateRBN(A_eps_8(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon8 = [0.8];
verify_time8 = [sum(vt8)];
safe8 = [sum(rb8==1)];
unsafe8 = [sum(rb8 == 0)];
unknown8 = [sum(rb8 == 2)];
T8 = table(epsilon8, safe8, unsafe8, unknown8, verify_time8)
fprintf('total time absdom (eps=0.8): %f ',verify_time8);
save("tanh_absdom_eps_8_verify.mat", 'T8', 'r8', 'rb8', 'cE8', 'cands8', 'vt8');

[r9, rb9, cE9, cands9, vt9] = nnv_net.evaluateRBN(A_eps_9(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon9 = [0.9];
verify_time9 = [sum(vt9)];
safe9 = [sum(rb9==1)];
unsafe9 = [sum(rb9 == 0)];
unknown9 = [sum(rb9 == 2)];
T9 = table(epsilon9, safe9, unsafe9, unknown9, verify_time9)
fprintf('total time absdom (eps=0.9): %f ',verify_time9);
save("tanh_absdom_eps_9_verify.mat", 'T9', 'r9', 'rb9', 'cE9', 'cands9', 'vt9');

[r10, rb10, cE10, cands10, vt10] = nnv_net.evaluateRBN(A_eps_10(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon10 = [1.0];
verify_time10 = [sum(vt10)];
safe10 = [sum(rb10==1)];
unsafe10 = [sum(rb10 == 0)];
unknown10 = [sum(rb10 == 2)];
T10 = table(epsilon10, safe10, unsafe10, unknown10, verify_time10)
fprintf('total time absdom (eps=0.10): %f ',verify_time10);
save("tanh_absdom_eps_10_verify.mat", 'T10', 'r10', 'rb10', 'cE10', 'cands10', 'vt10');



