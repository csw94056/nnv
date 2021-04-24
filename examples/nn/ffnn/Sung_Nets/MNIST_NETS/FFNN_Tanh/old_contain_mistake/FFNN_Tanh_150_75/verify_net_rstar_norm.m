close all;
clear;
clc;

load normalized/inputRStar_norm001.mat RS_eps_norm_001
load normalized/inputRStar_norm002.mat RS_eps_norm_002 
load normalized/inputRStar_norm003.mat RS_eps_norm_003 
load normalized/inputRStar_norm004.mat RS_eps_norm_004 
load normalized/inputRStar_norm005.mat RS_eps_norm_005 
load normalized/inputRStar_norm006.mat RS_eps_norm_006 
load normalized/inputRStar_norm007.mat RS_eps_norm_007 
load normalized/inputRStar_norm008.mat RS_eps_norm_008 
load normalized/inputRStar_norm009.mat RS_eps_norm_009
load normalized/inputRStar_norm010.mat RS_eps_norm_010 
load normalized/inputRStar_norm015.mat RS_eps_norm_015 
load normalized/inputRStar_norm020.mat RS_eps_norm_020
load normalized/inputRStar_norm025.mat RS_eps_norm_025 
load normalized/inputRStar_norm030.mat RS_eps_norm_030

load MNIST_tanh_150_75_normalized_DenseNet.mat net;
L1 = LayerS(net.Layers(3).Weights, net.Layers(3).Bias, 'tansig');
L2 = LayerS(net.Layers(5).Weights, net.Layers(5).Bias, 'tansig');
L3 = LayerS(net.Layers(7).Weights, net.Layers(7).Bias, 'purelin');
nnv_net = FFNNS([L1 L2 L3]);

load images_normalized.mat;
labels = IM_labels;

figure                                          % initialize figure
colormap(gray)                                  % set to grayscale
for i = 1:50                                    % preview first 50 samples
    subplot(5,10,i);                              % plot them in 6 x 6 grid
    digit = reshape(RS_eps_norm_005(i).lb{1}, [28,28]);     % row = 28 x 28 image
    imagesc(digit);                              % show the image
    title(labels(i));                   % show the label
end

N = 50; 
numCores = 1;
reachMethod = 'rstar-absdom-two';

[r001, rb001, cE001, cands001, vt001] = nnv_net.evaluateRBN(RS_eps_norm_001(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon001 = [0.001];
verify_time001 = [sum(vt001)];
safe001 = [sum(rb001==1)];
unsafe001 = [sum(rb001 == 0)];
unknown001 = [sum(rb001 == 2)];
T001 = table(epsilon001, safe001, unsafe001, unknown001, verify_time001)
fprintf('total time rstar norm (eps=0.001): %f ',verify_time001);
save("tanh_rstar_eps_001_verify_norm.mat", 'T001', 'r001', 'rb001', 'cE001', 'cands001', 'vt001');


[r002, rb002, cE002, cands002, vt002] = nnv_net.evaluateRBN(RS_eps_norm_002(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon002 = [0.002];
verify_time002 = [sum(vt002)];
safe002 = [sum(rb002==1)];
unsafe002 = [sum(rb002 == 0)];
unknown002 = [sum(rb002 == 2)];
T002 = table(epsilon002, safe002, unsafe002, unknown002, verify_time002)
fprintf('total time rstar norm (eps=0.002): %f ',verify_time002);
save("tanh_rstar_eps_002_verify_norm.mat", 'T002', 'r002', 'rb002', 'cE002', 'cands002', 'vt002');


[r003, rb003, cE003, cands003, vt003] = nnv_net.evaluateRBN(RS_eps_norm_003(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon003 = [0.003];
verify_time003 = [sum(vt003)];
safe003 = [sum(rb003==1)];
unsafe003 = [sum(rb003 == 0)];
unknown003 = [sum(rb003 == 2)];
T003 = table(epsilon003, safe003, unsafe003, unknown003, verify_time003)
fprintf('total time rstar norm (eps=0.003): %f ',verify_time003);
save("tanh_rstar_eps_003_verify_norm.mat", 'T003', 'r003', 'rb003', 'cE003', 'cands003', 'vt003');


[r004, rb004, cE004, cands004, vt004] = nnv_net.evaluateRBN(RS_eps_norm_004(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon004 = [0.004];
verify_time004 = [sum(vt004)];
safe004 = [sum(rb004==1)];
unsafe004 = [sum(rb004 == 0)];
unknown004 = [sum(rb004 == 2)];
T004 = table(epsilon004, safe004, unsafe004, unknown004, verify_time004)
fprintf('total time rstar norm (eps=0.004): %f ',verify_time004);
save("tanh_rstar_eps_004_verify_norm.mat", 'T004', 'r004', 'rb004', 'cE004', 'cands004', 'vt004');


[r005, rb005, cE005, cands005, vt005] = nnv_net.evaluateRBN(RS_eps_norm_005(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon005 = [0.005];
verify_time005 = [sum(vt005)];
safe005 = [sum(rb005==1)];
unsafe005 = [sum(rb005 == 0)];
unknown005 = [sum(rb005 == 2)];
T005 = table(epsilon005, safe005, unsafe005, unknown005, verify_time005)
fprintf('total time rstar norm (eps=0.005): %f ',verify_time005);
save("tanh_rstar_eps_005_verify_norm.mat", 'T005', 'r005', 'rb005', 'cE005', 'cands005', 'vt005');


[r006, rb006, cE006, cands006, vt006] = nnv_net.evaluateRBN(RS_eps_norm_006(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon006 = [0.006];
verify_time006 = [sum(vt006)];
safe006 = [sum(rb006==1)];
unsafe006 = [sum(rb006 == 0)];
unknown006 = [sum(rb006 == 2)];
T006 = table(epsilon006, safe006, unsafe006, unknown006, verify_time006)
fprintf('total time rstar norm (eps=0.006): %f ',verify_time006);
save("tanh_rstar_eps_006_verify_norm.mat", 'T006', 'r006', 'rb006', 'cE006', 'cands006', 'vt006');


[r007, rb007, cE007, cands007, vt007] = nnv_net.evaluateRBN(RS_eps_norm_007(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon007 = [0.007];
verify_time007 = [sum(vt007)];
safe007 = [sum(rb007 == 1)];
unsafe007 = [sum(rb007 == 0)];
unknown007 = [sum(rb007 == 2)];
T007 = table(epsilon007, safe007, unsafe007, unknown007, verify_time007)
fprintf('total time rstar norm (eps=0.007): %f ',verify_time007);
save("tanh_rstar_eps_007_verify_norm.mat", 'T007', 'r007', 'rb007', 'cE007', 'cands007', 'vt007');


[r008, rb008, cE008, cands008, vt008] = nnv_net.evaluateRBN(RS_eps_norm_008(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon008 = [0.008];
verify_time008 = [sum(vt008)];
safe008 = [sum(rb008==1)];
unsafe008 = [sum(rb008 == 0)];
unknown008 = [sum(rb008 == 2)];
T008 = table(epsilon008, safe008, unsafe008, unknown008, verify_time008)
fprintf('total time rstar norm (eps=0.008): %f ',verify_time008);
save("tanh_rstar_eps_008_verif_normy.mat", 'T008', 'r008', 'rb008', 'cE008', 'cands008', 'vt008');


[r009, rb009, cE009, cands009, vt009] = nnv_net.evaluateRBN(RS_eps_norm_009(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon009 = [0.009];
verify_time009 = [sum(vt009)];
safe009 = [sum(rb009 == 1)];
unsafe009 = [sum(rb009 == 0)];
unknown009 = [sum(rb009 == 2)];
T009 = table(epsilon009, safe009, unsafe009, unknown009, verify_time009)
fprintf('total time rstar norm (eps=0.009): %f ',verify_time009);
save("tanh_rstar_eps_009_verify_norm.mat", 'T009', 'r009', 'rb009', 'cE009', 'cands009', 'vt009');

[r010, rb010, cE010, cands010, vt010] = nnv_net.evaluateRBN(RS_eps_norm_010(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon010 = [0.010];
verify_time010 = [sum(vt010)];
safe010 = [sum(rb010==1)];
unsafe010 = [sum(rb010 == 0)];
unknown010 = [sum(rb010 == 2)];
T010 = table(epsilon010, safe010, unsafe010, unknown010, verify_time010)
fprintf('total time rstar norm (eps=0.010): %f ',verify_time010);
save("tanh_rstar_eps_010_verify_norm.mat", 'T010', 'r010', 'rb010', 'cE010', 'cands010', 'vt010');

[r015, rb015, cE015, cands015, vt015] = nnv_net.evaluateRBN(RS_eps_norm_015(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon015 = [0.015];
verify_time015 = [sum(vt015)];
safe015 = [sum(rb015==1)];
unsafe015 = [sum(rb015 == 0)];
unknown015 = [sum(rb015 == 2)];
T015 = table(epsilon015, safe015, unsafe015, unknown015, verify_time015)
fprintf('total time rstar norm (eps=0.015): %f ',verify_time015);
save("tanh_rstar_eps_015_verify_norm.mat", 'T015', 'r015', 'rb015', 'cE015', 'cands015', 'vt015');

[r020, rb020, cE020, cands020, vt020] = nnv_net.evaluateRBN(RS_eps_norm_020(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon020 = [0.020];
verify_time020 = [sum(vt020)];
safe020 = [sum(rb020==1)];
unsafe020 = [sum(rb020 == 0)];
unknown020 = [sum(rb020 == 2)];
T020 = table(epsilon020, safe020, unsafe020, unknown020, verify_time020)
fprintf('total time rstar norm (eps=0.020): %f ',verify_time020);
save("tanh_rstar_eps_020_verify_norm.mat", 'T020', 'r020', 'rb020', 'cE020', 'cands020', 'vt020');

[r025, rb025, cE025, cands025, vt025] = nnv_net.evaluateRBN(RS_eps_norm_025(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon025 = [0.025];
verify_time025 = [sum(vt025)];
safe025 = [sum(rb025==1)];
unsafe025 = [sum(rb025 == 0)];
unknown025 = [sum(rb025 == 2)];
T025 = table(epsilon025, safe025, unsafe025, unknown025, verify_time025)
fprintf('total time rstar norm (eps=0.025): %f ',verify_time025);
save("tanh_rstar_eps_025_verify_norm.mat", 'T025', 'r025', 'rb025', 'cE025', 'cands025', 'vt025');

[r030, rb030, cE030, cands030, vt030] = nnv_net.evaluateRBN(RS_eps_norm_030(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon030 = [0.030];
verify_time030 = [sum(vt030)];
safe030 = [sum(rb030==1)];
unsafe030 = [sum(rb030 == 0)];
unknown030 = [sum(rb030 == 2)];
T030 = table(epsilon030, safe030, unsafe030, unknown030, verify_time030)
fprintf('total time rstar norm (eps=0.030): %f ',verify_time030);
save("tanh_rstar_eps_030_verify_norm.mat", 'T030', 'r030', 'rb030', 'cE030', 'cands030', 'vt030');

