% close all;
% clear;
% clc;

load normalized/inputRStar_norm002.mat;
load normalized/inputRStar_norm004.mat;
load normalized/inputRStar_norm006.mat;
load normalized/inputRStar_norm008.mat;
load normalized/inputRStar_norm010.mat;
load normalized/inputRStar_norm012.mat;
load normalized/inputRStar_norm014.mat;
load normalized/inputRStar_norm016.mat;
load normalized/inputRStar_norm018.mat;
load normalized/inputRStar_norm020.mat;

load MNIST_tanh_100_100_normalized_DenseNet.mat net;
L1 = LayerS(net.Layers(3).Weights, net.Layers(3).Bias, 'logsig');
L2 = LayerS(net.Layers(5).Weights, net.Layers(5).Bias, 'logsig');
L3 = LayerS(net.Layers(7).Weights, net.Layers(7).Bias, 'purelin');
nnv_net = FFNNS([L1 L2 L3]);

load images_normalized.mat;
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
RS = RStar(S_eps_norm_002(1:N));
% [r002, rb002, cE002, cands002, vt002] = nnv_net.evaluateRBN(S_eps_norm_002(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
[r002, rb002, cE002, cands002, vt002] = nnv_net.evaluateRBN(RS(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon002 = [0.02];
verify_time002 = [sum(vt002)];
safe002 = [sum(rb002==1)];
unsafe002 = [sum(rb002 == 0)];
unknown002 = [sum(rb002 == 2)];
T002 = table(epsilon002, safe002, unsafe002, unknown002, verify_time002)
fprintf('total time rstar norm (eps=0.02): %f ',verify_time002);
save("verify_result/sigmoid_rstar_eps_002_verify_norm.mat", 'T002', 'r002', 'rb002', 'cE002', 'cands002', 'vt002');

[r004, rb004, cE004, cands004, vt004] = nnv_net.evaluateRBN(RS_eps_norm_004(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon004 = [0.04];
verify_time004 = [sum(vt004)];
safe004 = [sum(rb004==1)];
unsafe004 = [sum(rb004 == 0)];
unknown004 = [sum(rb004 == 2)];
T004 = table(epsilon004, safe004, unsafe004, unknown004, verify_time004)
fprintf('total time rstar norm (eps=0.04): %f ',verify_time004);
save("verify_result/sigmoid_rstar_eps_004_verify_norm.mat", 'T004', 'r004', 'rb004', 'cE004', 'cands004', 'vt004');

[r006, rb006, cE006, cands006, vt006] = nnv_net.evaluateRBN(RS_eps_norm_006(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon006 = [0.06];
verify_time006 = [sum(vt006)];
safe006 = [sum(rb006==1)];
unsafe006 = [sum(rb006 == 0)];
unknown006 = [sum(rb006 == 2)];
T006 = table(epsilon006, safe006, unsafe006, unknown006, verify_time006)
fprintf('total time rstar norm (eps=0.06): %f ',verify_time006);
save("verify_result/sigmoid_rstar_eps_006_verify_norm.mat", 'T006', 'r006', 'rb006', 'cE006', 'cands006', 'vt006');

[r008, rb008, cE008, cands008, vt008] = nnv_net.evaluateRBN(RS_eps_norm_008(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon008 = [0.08];
verify_time008 = [sum(vt008)];
safe008 = [sum(rb008==1)];
unsafe008 = [sum(rb008 == 0)];
unknown008 = [sum(rb008 == 2)];
T008 = table(epsilon008, safe008, unsafe008, unknown008, verify_time008)
fprintf('total time rstar norm (eps=0.08): %f ',verify_time008);
save("verify_result/sigmoid_rstar_eps_008_verify_norm.mat", 'T008', 'r008', 'rb008', 'cE008', 'cands008', 'vt008');

[r010, rb010, cE010, cands010, vt010] = nnv_net.evaluateRBN(RS_eps_norm_010(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon010 = [0.10];
verify_time010 = [sum(vt010)];
safe010 = [sum(rb010==1)];
unsafe010 = [sum(rb010 == 0)];
unknown010 = [sum(rb010 == 2)];
T010 = table(epsilon010, safe010, unsafe010, unknown010, verify_time010)
fprintf('total time rstar norm (eps=0.10): %f ',verify_time010);
save("verify_result/sigmoid_rstar_eps_010_verify_norm.mat", 'T010', 'r010', 'rb010', 'cE010', 'cands010', 'vt010');

[r012, rb012, cE012, cands012, vt012] = nnv_net.evaluateRBN(RS_eps_norm_012(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon012 = [0.12];
verify_time012 = [sum(vt012)];
safe012 = [sum(rb012==1)];
unsafe012 = [sum(rb012 == 0)];
unknown012 = [sum(rb012 == 2)];
T012 = table(epsilon012, safe012, unsafe012, unknown012, verify_time012)
fprintf('total time rstar norm (eps=0.12): %f ',verify_time012);
save("verify_result/sigmoid_rstar_eps_012_verify_norm.mat", 'T012', 'r012', 'rb012', 'cE012', 'cands012', 'vt012');

[r014, rb014, cE014, cands014, vt014] = nnv_net.evaluateRBN(RS_eps_norm_014(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon014 = [0.14];
verify_time014 = [sum(vt014)];
safe014 = [sum(rb014==1)];
unsafe014 = [sum(rb014 == 0)];
unknown014 = [sum(rb014 == 2)];
T014 = table(epsilon014, safe014, unsafe014, unknown014, verify_time014)
fprintf('total time rstar norm (eps=0.14): %f ',verify_time014);
save("verify_result/sigmoid_rstar_eps_014_verify_norm.mat", 'T014', 'r014', 'rb014', 'cE014', 'cands014', 'vt014');

[r016, rb016, cE016, cands016, vt016] = nnv_net.evaluateRBN(RS_eps_norm_016(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon016 = [0.16];
verify_time016 = [sum(vt016)];
safe016 = [sum(rb016==1)];
unsafe016 = [sum(rb016 == 0)];
unknown016 = [sum(rb016 == 2)];
T016 = table(epsilon016, safe016, unsafe016, unknown016, verify_time016)
fprintf('total time rstar norm (eps=0.16): %f ',verify_time016);
save("verify_result/sigmoid_rstar_eps_016_verify_norm.mat", 'T016', 'r016', 'rb016', 'cE016', 'cands016', 'vt016');

[r018, rb018, cE018, cands018, vt018] = nnv_net.evaluateRBN(RS_eps_norm_018(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon018 = [0.18];
verify_time018 = [sum(vt018)];
safe018 = [sum(rb018==1)];
unsafe018 = [sum(rb018 == 0)];
unknown018 = [sum(rb018 == 2)];
T018 = table(epsilon018, safe018, unsafe018, unknown018, verify_time018)
fprintf('total time rstar norm (eps=0.18): %f ',verify_time018);
save("verify_result/sigmoid_rstar_eps_018_verify_norm.mat", 'T018', 'r018', 'rb018', 'cE018', 'cands018', 'vt018');

[r020, rb020, cE020, cands020, vt020] = nnv_net.evaluateRBN(RS_eps_norm_020(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
epsilon020 = [0.20];
verify_time020 = [sum(vt020)];
safe020 = [sum(rb020==1)];
unsafe020 = [sum(rb020 == 0)];
unknown020 = [sum(rb020 == 2)];
T020 = table(epsilon020, safe020, unsafe020, unknown020, verify_time020)
fprintf('total time rstar norm (eps=0.20): %f ',verify_time020);
save("verify_result/sigmoid_rstar_eps_020_verify_norm.mat", 'T020', 'r020', 'rb020', 'cE020', 'cands020', 'vt020');

% [r02, rb02, cE02, cands02, vt02] = nnv_net.evaluateRBN(RS_eps_norm_02(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon02 = [0.2];
% verify_time02 = [sum(vt02)];
% safe02 = [sum(rb02==1)];
% unsafe02 = [sum(rb02 == 0)];
% unknown02 = [sum(rb02 == 2)];
% T02 = table(epsilon02, safe02, unsafe02, unknown02, verify_time02)
% fprintf('total time rstar norm (eps=0.2): %f ',verify_time02);
% save("verify_result/sigmoid_rstar_eps_02_verify_norm.mat", 'T02', 'r02', 'rb02', 'cE02', 'cands02', 'vt02');
% 
% [r04, rb04, cE04, cands04, vt04] = nnv_net.evaluateRBN(RS_eps_norm_04(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon04 = [0.4];
% verify_time04 = [sum(vt04)];
% safe04 = [sum(rb04==1)];
% unsafe04 = [sum(rb04 == 0)];
% unknown04 = [sum(rb04 == 2)];
% T04 = table(epsilon04, safe04, unsafe04, unknown04, verify_time04)
% fprintf('total time rstar norm (eps=0.4): %f ',verify_time04);
% save("verify_result/sigmoid_rstar_eps_04_verify_norm.mat", 'T04', 'r04', 'rb04', 'cE04', 'cands04', 'vt04');
% 
% [r06, rb06, cE06, cands06, vt06] = nnv_net.evaluateRBNR(S_eps_norm_06(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon06 = [0.6];
% verify_time06 = [sum(vt06)];
% safe06 = [sum(rb06==1)];
% unsafe06 = [sum(rb06 == 0)];
% unknown06 = [sum(rb06 == 2)];
% T06 = table(epsilon06, safe06, unsafe06, unknown06, verify_time06)
% fprintf('total time rstar norm (eps=0.6): %f ',verify_time06);
% save("verify_result/sigmoid_rstar_eps_06_verify_norm.mat", 'T06', 'r06', 'rb06', 'cE06', 'cands06', 'vt06');
% 
% [r08, rb08, cE08, cands08, vt08] = nnv_net.evaluateRBN(RS_eps_norm_08(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon08 = [0.8];
% verify_time08 = [sum(vt08)];
% safe08 = [sum(rb08==1)];
% unsafe08 = [sum(rb08 == 0)];
% unknown08 = [sum(rb08 == 2)];
% T08 = table(epsilon08, safe08, unsafe08, unknown08, verify_time08)
% fprintf('total time rstar norm (eps=0.8): %f ',verify_time08);
% save("verify_result/sigmoid_rstar_eps_08_verify_norm.mat", 'T08', 'r08', 'rb08', 'cE08', 'cands08', 'vt08');
% 
% [r10, rb10, cE10, cands10, vt10] = nnv_net.evaluateRBN(RS_eps_norm_10(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon10 = [1.0];
% verify_time10 = [sum(vt10)];
% safe10 = [sum(rb10==1)];
% unsafe10 = [sum(rb10 == 0)];
% unknown10 = [sum(rb10 == 2)];
% T10 = table(epsilon10, safe10, unsafe10, unknown10, verify_time10)
% fprintf('total time rstar norm (eps=1.0): %f ',verify_time10);
% save("verify_result/sigmoid_rstar_eps_10_verify_norm.mat", 'T10', 'r10', 'rb10', 'cE10', 'cands10', 'vt10');
% 
% [r12, rb12, cE12, cands12, vt12] = nnv_net.evaluateRBN(RS_eps_norm_12(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon12 = [1.2];
% verify_time12 = [sum(vt12)];
% safe12 = [sum(rb12==1)];
% unsafe12 = [sum(rb12 == 0)];
% unknown12 = [sum(rb12 == 2)];
% T12 = table(epsilon12, safe12, unsafe12, unknown12, verify_time12)
% fprintf('total time rstar norm (eps=1.2): %f ',verify_time12);
% save("verify_result/sigmoid_rstar_eps_12_verify_norm.mat", 'T12', 'r12', 'rb12', 'cE12', 'cands12', 'vt12');
% 
% [r14, rb14, cE14, cands14, vt14] = nnv_net.evaluateRBN(RS_eps_norm_14(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon14 = [1.4];
% verify_time14 = [sum(vt14)];
% safe14 = [sum(rb14==1)];
% unsafe14 = [sum(rb14 == 0)];
% unknown14 = [sum(rb14 == 2)];
% T14 = table(epsilon14, safe14, unsafe14, unknown14, verify_time14)
% fprintf('total time rstar norm (eps=1.4): %f ',verify_time14);
% save("verify_result/sigmoid_rstar_eps_14_verify_norm.mat", 'T14', 'r14', 'rb14', 'cE14', 'cands14', 'vt14');
% 
% [r16, rb16, cE16, cands16, vt16] = nnv_net.evaluateRBN(RS_eps_norm_16(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon16 = [1.6];
% verify_time16 = [sum(vt16)];
% safe16 = [sum(rb16==1)];
% unsafe16 = [sum(rb16 == 0)];
% unknown16 = [sum(rb16 == 2)];
% T16 = table(epsilon16, safe16, unsafe16, unknown16, verify_time16)
% fprintf('total time rstar norm (eps=1.6): %f ',verify_time16);
% save("verify_result/sigmoid_rstar_eps_16_verify_norm.mat", 'T16', 'r16', 'rb16', 'cE16', 'cands16', 'vt16');
% 
% [r18, rb18, cE18, cands18, vt18] = nnv_net.evaluateRBN(RS_eps_norm_18(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon18 = [1.8];
% verify_time18 = [sum(vt18)];
% safe18 = [sum(rb18==1)];
% unsafe18 = [sum(rb18 == 0)];
% unknown18 = [sum(rb18 == 2)];
% T18 = table(epsilon18, safe18, unsafe18, unknown18, verify_time18)
% fprintf('total time rstar norm (eps=1.8): %f ',verify_time18);
% save("verify_result/sigmoid_rstar_eps_18_verify_norm.mat", 'T18', 'r18', 'rb18', 'cE18', 'cands18', 'vt18');
% 
% [r20, rb20, cE20, cands20, vt20] = nnv_net.evaluateRBN(RS_eps_norm_20(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon20 = [2.0];
% verify_time20 = [sum(vt20)];
% safe20 = [sum(rb20==1)];
% unsafe20 = [sum(rb20 == 0)];
% unknown20 = [sum(rb20 == 2)];
% T20 = table(epsilon20, safe20, unsafe20, unknown20, verify_time20)
% fprintf('total time rstar norm (eps=2.0): %f ',verify_time20);
% save("verify_result/sigmoid_rstar_eps_20_verify_norm.mat", 'T20', 'r20', 'rb20', 'cE20', 'cands20', 'vt20');

% [r005, rb005, cE005, cands005, vt005] = nnv_net.evaluateRBN(RS_eps_norm_005(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon005 = [0.005];
% verify_time005 = [sum(vt005)];
% safe005 = [sum(rb005==1)];
% unsafe005 = [sum(rb005 == 0)];
% unknown005 = [sum(rb005 == 2)];
% T005 = table(epsilon005, safe005, unsafe005, unknown005, verify_time005)
% fprintf('total time rstar norm (eps=0.005): %f ',verify_time005);
% save("sigmoid_rstar_eps_005_verify_norm.mat", 'T005', 'r005', 'rb005', 'cE005', 'cands005', 'vt005');
% 
% [r010, rb010, cE010, cands010, vt010] = nnv_net.evaluateRBN(RS_eps_norm_010(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon010 = [0.010];
% verify_time010 = [sum(vt010)];
% safe010 = [sum(rb010==1)];
% unsafe010 = [sum(rb010 == 0)];
% unknown010 = [sum(rb010 == 2)];
% T010 = table(epsilon010, safe010, unsafe010, unknown010, verify_time010)
% fprintf('total time rstar norm (eps=0.010): %f ',verify_time010);
% save("sigmoid_rstar_eps_010_verify_norm.mat", 'T010', 'r010', 'rb010', 'cE010', 'cands010', 'vt010');
% 
% [r015, rb015, cE015, cands015, vt015] = nnv_net.evaluateRBN(RS_eps_norm_015(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon015 = [0.015];
% verify_time015 = [sum(vt015)];
% safe015 = [sum(rb015==1)];
% unsafe015 = [sum(rb015 == 0)];
% unknown015 = [sum(rb015 == 2)];
% T015 = table(epsilon015, safe015, unsafe015, unknown015, verify_time015)
% fprintf('total time rstar norm (eps=0.015): %f ',verify_time015);
% save("sigmoid_rstar_eps_015_verify_norm.mat", 'T015', 'r015', 'rb015', 'cE015', 'cands015', 'vt015');
% 
% [r020, rb020, cE020, cands020, vt020] = nnv_net.evaluateRBN(RS_eps_norm_020(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon020 = [0.020];
% verify_time020 = [sum(vt020)];
% safe020 = [sum(rb020==1)];
% unsafe020 = [sum(rb020 == 0)];
% unknown020 = [sum(rb020 == 2)];
% T020 = table(epsilon020, safe020, unsafe020, unknown020, verify_time020)
% fprintf('total time rstar norm (eps=0.020): %f ',verify_time020);
% save("sigmoid_rstar_eps_020_verify_norm.mat", 'T020', 'r020', 'rb020', 'cE020', 'cands020', 'vt020');
% 
% [r025, rb025, cE025, cands025, vt025] = nnv_net.evaluateRBN(RS_eps_norm_025(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon025 = [0.025];
% verify_time025 = [sum(vt025)];
% safe025 = [sum(rb025==1)];
% unsafe025 = [sum(rb025 == 0)];
% unknown025 = [sum(rb025 == 2)];
% T025 = table(epsilon025, safe025, unsafe025, unknown025, verify_time025)
% fprintf('total time rstar norm (eps=0.025): %f ',verify_time025);
% save("sigmoid_rstar_eps_025_verify_norm.mat", 'T025', 'r025', 'rb025', 'cE025', 'cands025', 'vt025');
% 
% [r030, rb030, cE030, cands030, vt030] = nnv_net.evaluateRBN(RS_eps_norm_030(1:N), labels(1:N)+1, reachMethod, numCores, 0, 0, 'glpk');
% epsilon030 = [0.030];
% verify_time030 = [sum(vt030)];
% safe030 = [sum(rb030==1)];
% unsafe030 = [sum(rb030 == 0)];
% unknown030 = [sum(rb030 == 2)];
% T030 = table(epsilon030, safe030, unsafe030, unknown030, verify_time030)
% fprintf('total time rstar norm (eps=0.030): %f ',verify_time030);
% save("sigmoid_rstar_eps_030_verify_norm.mat", 'T030', 'r030', 'rb030', 'cE030', 'cands030', 'vt030');

