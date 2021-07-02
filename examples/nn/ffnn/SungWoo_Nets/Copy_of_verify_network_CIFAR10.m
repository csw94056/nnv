close all;
clear;
clc;

%%
% network trained with images: [0 1] -> normalized, 
%                              [0 255] ->  not_normalized
% dataset_ = 'MNIST';
% net_ = 'MNIST_FNNsmall_sigmoid';
dataset_ = 'CIFAR10';
net_ = 'CIFAR10_FNNsmall_tanh';
n_ = 'FNNsmall';
normalized = 0;


norm_ = '';
if normalized
    norm_ = '_normalized'
end

net_dir = sprintf('%s/nets/%s/%s.mat', dataset_,n_,net_)
% image_dir = sprintf('data/%s.csv',net_);
% image_dir = sprintf('data/%s_100images.csv',net_);
% image_dir = sprintf('%s/data/%s%s_raw.csv',dataset_,net_, norm_)
image_dir = sprintf('%s/data/%s_raw.csv', dataset_,net_)


normalized = 1;
%% load network
load(net_dir);
nnv_net = net2nnv_net(net);

% classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
% net = importONNXNetwork('train/FNNbig/ffnnTANH__Point_6_500.onnx','OutputLayerType','classification','Classes',classes);
% % net = importONNXNetwork('train/FNNbig/ffnnTANH__PGDK_w_0.1_6_500.onnx','OutputLayerType','classification','Classes',classes);
% act_fn = 'tansig';
% L1 = LayerS(reshape(net.Layers(6).Weights, [784,500])', reshape(net.Layers(6).Bias, [1 500])', act_fn);
% L2 = LayerS(reshape(net.Layers(9).Weights, [500,500])', reshape(net.Layers(9).Bias, [1 500])', act_fn);
% L3 = LayerS(reshape(net.Layers(12).Weights, [500,500])', reshape(net.Layers(12).Bias, [1 500])', act_fn);
% L4 = LayerS(reshape(net.Layers(15).Weights, [500,500])', reshape(net.Layers(15).Bias, [1 500])', act_fn);
% L5 = LayerS(reshape(net.Layers(18).Weights, [500,500])', reshape(net.Layers(18).Bias, [1 500])', act_fn);
% L6 = LayerS(reshape(net.Layers(21).Weights, [500,500])', reshape(net.Layers(21).Bias, [1 500])', act_fn);
% L7 = LayerS(reshape(net.Layers(24).Weights, [500,10])', reshape(net.Layers(24).Bias, [1 10])', act_fn);
% nnv_net = FFNNS([L1 L2 L3 L4 L5 L6 L7]);

%% load images
csv_data = csvread(image_dir);
IM_labels = csv_data(:,1);
IM_data = csv_data(:,2:end)';

% reachMethod = 'approx-star';
reachMethod = 'rstar-absdom-two';
% reachMethod = 'absdom';

% reachMethod = 'abs-dom';
% reachMethod = 'approx-zono';

relaxFactor = [0];
numCores = 1;
disp_opt = 0;
lp_solver = 'linprog' % 'linprog'

%eps = [0, 1, 2, 3, 4, 5, 6, 7, 8];
% eps = [0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.024, 0.026, 0.028, 0.030];
% eps = [0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02];

% eps = [0.004,0.006,0.008,0.010,0.012,0.014];
% eps = [0.0020, 0.0030, 0.0040, 0.005, 0.006, 0.007, 0.008];

% eps = [0.0010, 0.0020, 0.0030, 0.0040, 0.005, 0.006, 0.007];
eps = [0.0010, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016];

% eps = [0.010, 0.012, 0.014, 0.016, 0.018, 0.0200];%, 0.022, 0.024, 0.026, 0.028, 0.030];
% eps = [0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.0020]
% eps = [0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015];
% eps = [0.01, 0.02, 0.03, 0.04, 0.05];
% eps = 0.02;
% eps = 0.0010

N = size(IM_data, 2);
K = length(relaxFactor);
M = length(eps);

r = zeros(K, M); % percentage of images that are robust
rb = cell(K, M); % detail robustness verification result
cE = cell(K, M); % detail counterexamples
vt = cell(K, M); % detail verification time
cands = cell(K,M); % counterexample
total_vt = zeros(K, M); % total verification time


% Tanh_100_100_eps030 = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25, ...
%     27,28,29,30,31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51,52,53] % tanh_100_100_eps030
% Tanh_100_100_eps020 = [1,2,4,5,6,7,10,11,13,14,15,16,17,18,20,22,23,24,25, ...
%     27,28,29,30,31,33,36,37,38,40,42,43,44,45,46,48,49,50,51,52,53,54,55,56,57,59,61,65,68,69,70] % tanh_100_100_eps020
% Tanh_100_50_eps020 = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,20,22,23,24,25,27,28,29,30,...
%     31,32,33,35,36,37,38,40,41,42,43,44,45,46,48,49,50,51,52,53,54,55,56,57];
% Sigmoid_100_50_eps020 = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,18,19,20,22,23,24,25,26,27,28,29,...
%     31,33,34,35,36,37,38,40,41,42,43,44,46,48,49,50,51,52,53,54,55,56,57,58];

% MNIST_FNNsmall_tanh_original = [1,2,4,5,6,7,10,11,13,14,15,16,17,18,20,22,23,24,25, 27,28,29,30,31,33,36,37,38,40,42,43,44,45,46,48,49,50,51,52,53,54,55,56,57,59,61,65,68,69,70, ...
%     80,81,82,83,84,85,86,87,88,89,90,91,92,94,95,96,98,99,100,101,102,103,104,105,106,107,108,109,110,111,113,114,115,117,118,120,121,123,124,125,127,128,129,130,131,132,133,134,135,136];
% RS_MNIST_FNNbig_tanh = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,...
%     26,27,28,29,30,31,32,33,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51,52,....
%     53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,....
%     78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102];
% Z_MNIST_FNNbig_tanh = [2,36,55,61,69,71,72,80,83,86,89,104,118,132,133,142,148,162,163,184,187,189,198,201,202,209,223,224,237,238,239,247,249,...
%     259,263,286,294,297,299,307,317,328,335,344,352,376,385,397,403,411,424,438,441,443,452,453,462,463,464,476,478,494,495,501,504,513,...
%     514,519,542,550,557,560,574,575,577,591,593,603,613,624,626,627,634,644,649,652,657,658,666,677,678,681,687,695,702,712,720,732,737,738];


% RS_tanh_med_150 = [2,11,26,29,31,33,52,55,70,72,83,86,103,128,129,130,133,137,...
%     148,149,162,163,166,184,187,189,193,195,201,202,209,217,237,243,247,259,...
%     271,272,278,286,295,297,298,310,312,328,344,348,352,357,376,381,383,385,...
%     403,408,425,438,441,443,450,451,452,460,462,463,475,476,478,494,501,514,...
%     519,526,527,528,547,553,574,582,591,593,603,695,702,720,732,733,738,743,...
%     752,763,767,781,783,786,793,795,797,800]; %     %RS_tanh_med_150
% 
% S = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,...
%      28,29,30,31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51,52,...
%      53,54,55,56,57,58,59,60,61,65,68,69,70,71,72,73,74,75,76,77,78,80,81,82,...
%      83,84,86,87,89,90,91,92,94,95,96,98,99,100,101,102,103,104,106,107,108,...
%      109,110,111,114,115,116]; %MNIST_FNNsmall_sigmoid

% S = [1,2,5,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,31,...
%     33,35,36,37,43,44,45,46,48,49,50,51,52,53,54,55,56,57,59,60,61,64,65,68,...
%     69,70,71,72,73,76,77,78,80,83,84,85,86,87,88,89,91,92,94,96,99,100,101,102,...
%     103,104,106,107,109,110,111,113,114,115,120,121,123,124,125,127,128,...
%     129,130,131,132,133,134,135,137,138,139]; %MNIST_FNNmed_sigmoid

% S = [1,2,3,4,5,6,8,10,11,12,16,17,18,20,23,24,25,26,27,...
%     28,29,31,33,34,35,37,38,40,42,44,45,46,48,49,50,51,54,55,...
%     57,58,59,60,61,65,68,69,70,71,72,73,76,77,82,83,84,...
%     85,86,87,88,89,90,91,92,94,96,100,101,102,103,104,106,108,...
%     110,114,115,121,123,127,128,129,130,131,132,133,138,139,140,...
%     141,142,143,144,145,148,149,151,153,154,155,156,157]; %MINST_FNNbig_sigmoid

%S = [1,2,4,5,6,7,10,11,13,14,15,16,17,18,20,22,23,24,25, 27,28,29,30,31,33,36,37,38,40,42,43,44,45,46,48,49,50,51,52,53,54,55,56,57,59,61,65,68,69,70, ...
%    80,81,82,83,84,85,86,87,88,89,90,91,92,95,96,98,99,100,101,102,103,104,105,106,107,108,109,110,111,113,114,115,117,118,120,121,123,124,125,127,128,129,130,131,132,133,134,135,136,137];
%   MNIST_FNNsmall_tanh

% disp_opt = 'display';
% S = [];
% j = 1;
% eps(j)
% for s = 1:N %N
%     s
%     images = attack_images(IM_data(:,s), eps(j), reachMethod, normalized); 
%     labels = IM_labels(s)+1;
%     [r, rb, cE, cands, vt] = nnv_net.evaluateRBN(images, labels, reachMethod, numCores, relaxFactor , disp_opt, lp_solver)
%     if  rb==1 && j == 1
%         fprintf('safe: %d\n', s);
%         S = [S s];
%     elseif rb ==2
%         fprintf('unsafe: %d\n', s);
%     end
%     
%     if length(S) == 1
%         break;
%     end
% end
% S

%     %RS_tanh_med_150



% IM = [IM_labels(S) IM_data(:,S)'];
% writematrix(IM,'MNIST/data/MNIST_FNNbig_tanh.csv');
% 

S = [];
j = 1;
eps(j)

% images = attack_images(IM_data(:,1:300), eps(j), reachMethod, normalized); 
% labels = IM_labels;
% for s = 1:N%N
%     s
%  
%     [r, rb, cE, cands, vt] = nnv_net.evaluateRBN(images(s), labels(s), reachMethod, numCores, relaxFactor , disp_opt, lp_solver);
%     if  rb==1 && j == 1
%         fprintf('safe: %d\n', s);
%         S = [S s];
%     elseif rb ==2
%         fprintf('unsafe: %d\n', s);
%     end
%     
%     if length(S) == 150
%         break;
%     end
% end
% S


% images1 = attack_images(IM_data(:,1:100), eps(j), reachMethod, normalized); 
% labels1 = IM_labels(1:100)+1;
% dir = sprintf('%s1.mat',net_);
% save(dir,'images1','labels1');
% 
% images2 = attack_images(IM_data(:,101:200), eps(j), reachMethod, normalized); 
% labels2 = IM_labels(101:200)+1;
% dir = sprintf('%s2.mat',net_);
% save(dir,'images2','labels2');
% 
% images3 = attack_images(IM_data(:,201:300), eps(j), reachMethod, normalized); 
% labels3 = IM_labels(201:300)+1;
% dir = sprintf('%s3.mat',net_);
% save(dir,'images3','labels3');

% load CIFAR10_FNNsmall_tanh1.mat
% images = images1;
% labels = labels1-1;
% N = length(labels);
%  
% for s = 1:N
%     s
%     images = attack_images(IM_data(:,s), eps(j), reachMethod, normalized); 
%     labels = IM_labels(s) + 1;
%     
%     [r, rb, cE, cands, vt] = nnv_net.evaluateRBN(images, labels, reachMethod, numCores, relaxFactor , disp_opt, lp_solver);
%     if  rb==1 && j == 1
%         fprintf('safe: %d\n', s);
%         S = [S s];
%     elseif rb ==2
%         fprintf('unsafe: %d\n', s);
%     end
%     
%     if length(S) == 300
%         break;
%     end
% end
% S
% fprintf('\n[');fprintf('%d,',S);

% IM = [IM_labels(S) IM_data(:,S)'];
% writematrix(IM,'CIFAR10/data/CIFAR10_FNNbig_tanh.csv');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            CIFAR10 FNNsmall tanh
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% All = [3,6,12,15,19,20,30,35,55,82,93,100,103,108,113,117,131,137,155,162,167,...
%     174,176,180,187,190,194,199,203,218,222,236,256,266,290,293,297,298,299,...
%     301,302,312,332,334,335,339,340,348,365,372,380,387,391,393,407,420,436,...
%     448,456,473,478,482,487,490,491,494,496,512,517,520,535,537,543,546,551,...
%     557,572,573,577,579,591,594,603,610,611,613,618,633,658,661,662,663,664,...
%     668,671,682,694,697,701,706,712,714,722,733,744,748,751,753,757,760,775,...
%     778,783,786,795,801,804,813,816,824,841,843,854,859,865,880,881,886,889,...
%     921,922,935,943,944,947,952,954,956,960,962,964,965,968,978,986,993,995,...
%     1000,1004,1011];
% % S = All(1:100);
% % S = [3,6,12,20,30,55,82,93,100,103,108,113,117,137,162,...
% %     174,176,180,187,190,199,203,218,256,266,290,297,298,...
% %     301,302,312,332,334,335,339,340,365,372,380,393,407,420,...
% %     448,456,473,478,482,487,490,494,496,512,517,520,535,537,543,546,551,...
% %     572,577,579,591,594,603,610,611,613,618,633,658,661,662,663,664,...
% %     668,671,682,697,701,706,712,714,722,733,744];
% % Star = [3,6,12,20,30,55,82,93,100,103,108,113,117,137,162,174,176,180,187,...
% %     190,199,203,218,256,266,290,297,298,301,302,312,332,334,335,339,340,365,...
% %     372,380,393,407,420,448,456,473,478,482,487,490,494,496,512,517,520,535,...
% %     537,543,546,551,577,579,591,594,603,610,611,633,658,661,662,663,664,668,...
% %     671,682,697,701,706,712,714,722,733,744,748,751,753,757,760,775,783,786,...
% %     795,801,804,813,816,843,854,865,880]; %CIFAR10_FNNsmall_tanh
% % Extra_star = [748,751,753,757,760,775,783,786,795,801,804,813,816,843,854,...
% %     865,880,881,886,889,921];
% 
% S = [3,6,12,30,55,82,93,100,103,108,113,117,137,162,176,180,187,...
%     190,199,203,218,256,266,290,297,298,301,302,312,332,334,335,339,340,365,...
%     372,380,393,407,420,448,456,473,478,482,487,490,494,496,512,517,520,535,...
%     537,543,546,551,577,579,591,603,610,611,633,658,661,662,663,664,668,...
%     671,682,697,701,706,712,714,722,733,744,748,751,753,757,760,775,783,786,...
%     795,801,804,813,816,843,854,865,880,881,886,889]; %CIFAR10_FNNsmall_tanh
% 
% % S = [881,886,889,921];
% IM = [IM_labels(S) IM_data(:,S)'];
% writematrix(IM,'CIFAR10/data/CIFAR10_FNNsmall_tanh.csv');
% 
% % for i=1:K
% %     for j=1:M
% %         eps(j)
% %         images = attack_images(IM_data(:,S), eps(j), reachMethod, normalized); 
% %         labels = IM_labels(S)+1;
% %         t = tic;
% %         [r(i,j), rb{i,j}, cE{i,j}, cands{i,j}, vt{i,j}] = nnv_net.evaluateRBN(images, labels, reachMethod, numCores, relaxFactor , disp_opt, lp_solver);
% %         total_vt(i,j) = toc(t);
% %     end
% % end
% 
% disp_opt = 'display';
% numCores = 16
% start = 1
% for j=1:M
%     fprintf('\tepsilon: %f\n',eps(j)');
%     for k = start:length(S)
%         k
%         image = attack_images(IM_data(:,S(k)), eps(j), reachMethod, normalized); 
%         label = IM_labels(S(k)) + 1;
%         fprintf('S(k): %d\n',S(k));
% %         [r(i,j), rb{i,j}, cE{i,j}, cands{i,j}, vt{i,j}] = nnv_net.evaluateRBN(images(k), labels(k), reachMethod, numCores, relaxFactor , disp_opt, lp_solver)
%         [r, rb, cE, cands, vt] = nnv_net.evaluateRBN(image, label, reachMethod, numCores, relaxFactor , disp_opt, lp_solver)
%     end
%     start = 1;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %                            CIFAR10 FNNmed tanh
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % All = [3,30,51,55,67,74,93,117,131,137,155,194,226,236,245,291,298,299,311,...
% %     339,345,349,363,380,391,396,407,432,440,448,490,494,512,517,525,537,543,...
% %     545,572,575,585,610,611,613,661,663,664,668,671,681,682,694,714,722,744,...
% %     747,758,775,807,824,841,843,853,855,872,881,886,918,935,943,947,952,954,...
% %     968,978,993,995,1001,1011,1019,1030,1068,1077,1078,1079,1081,1086,1092,...
% %     1097,1103,1104,1120,1142,1143,1144,1161,1163,1201,1215,1238,1282,1296,...
% %     1302,1304,1312,1315,1317,1321,1328,1337,1341,1365,1366,1367,1370,1398,...
% %     1409,1434,1449,1450,1474,1489,1509,1520,1541,1556,1570,1580,1590,1615,...
% %     1618,1642,1651,1652,1668,1670,1680,1681,1692,1707,1715,1723,1727,1730,...
% %     1743,1744,1749,1755,1758,1777,1779,1782,1786,1796,1840,1852,1857,1865,...
% %     1867,1870,1876,1897,1913,1918,1950,1960,1974,1989,1990,2008,2013,2021,...
% %     2022,2029,2036,2042,2047,2049,2051,2053,2061,2080,2085,2090,2125,2140,...
% %     2151,2191,2201,2212,2215,2218,2226,2240,2242,2266,2267,2268,2275,2278];
% 
% S = [3,55,74,93,117,131,137,236,245,298,299,311,...
%     339,349,363,407,432,448,490,517,525,537,543,...
%     545,572,575,610,663,668,671,681,694,722,744,...
%     758,775,807,841,855,881,886,918,935,943,947,952,954,...
%     968,993,1001,1011,1019,1068,1077,1078,1079,1081,1092,...
%     1097,1103,1104,1120,1142,1143,1144,1161,1163,1201,1215,1238,....
%     1282,1296,1304,1312,1315,1321,1337,1341,1365,1366,1398,...
%     1434,1449,1474,1489,1509,1556,1580,1615,...
%     1618,1651,1652,1680,1681,1692,1715,1727,...
%     1744,1749,1755,]; %CIFAR10_FNNmed_tanh
% 
% Extra = [1777,1779];
% 
% % % Star = [3,6,12,20,30,55,82,93,100,103,108,113,117,137,162,174,176,180,187,...
% % %     190,199,203,218,256,266,290,297,298,301,302,312,332,334,335,339,340,365,...
% % %     372,380,393,407,420,448,456,473,478,482,487,490,494,496,512,517,520,535,...
% % %     537,543,546,551,577,579,591,594,603,610,611,633,658,661,662,663,664,668,...
% % %     671,682,697,701,706,712,714,722,733,744,748,751,753,757,760,775,783,786,...
% % %     795,801,804,813,816,843,854,865,880]; %CIFAR10_FNNsmall_tanh
% % % Extra_star = [748,751,753,757,760,775,783,786,795,801,804,813,816,843,854,...
% % %     865,880,881,886,889,921];
% % 
% % S = [3,6,12,30,55,82,93,100,103,108,113,117,137,162,176,180,187,...
% %     190,199,203,218,256,266,290,297,298,301,302,312,332,334,335,339,340,365,...
% %     372,380,393,407,420,448,456,473,478,482,487,490,494,496,512,517,520,535,...
% %     537,543,546,551,577,579,591,603,610,611,633,658,661,662,663,664,668,...
% %     671,682,697,701,706,712,714,722,733,744,748,751,753,757,760,775,783,786,...
% %     795,801,804,813,816,843,854,865,880,881,886,889]; %CIFAR10_FNNsmall_tanh
% % 
% % % S = [881,886,889,921];
% IM = [IM_labels(S) IM_data(:,S)'];
% writematrix(IM,'CIFAR10/data/CIFAR10_FNNmed_tanh.csv');
% % 
% % % for i=1:K
% % %     for j=1:M
% % %         eps(j)
% % %         images = attack_images(IM_data(:,S), eps(j), reachMethod, normalized); 
% % %         labels = IM_labels(S)+1;
% % %         t = tic;
% % %         [r(i,j), rb{i,j}, cE{i,j}, cands{i,j}, vt{i,j}] = nnv_net.evaluateRBN(images, labels, reachMethod, numCores, relaxFactor , disp_opt, lp_solver);
% % %         total_vt(i,j) = toc(t);
% % %     end
% % % end
% % 
% disp_opt = 'display';
% numCores = 26
% start = 11
% for j=6:M
%     fprintf('\tepsilon: %f\n',eps(j)');
%     for k = start:length(S)
%         k
%         image = attack_images(IM_data(:,S(k)), eps(j), reachMethod, normalized); 
%         label = IM_labels(S(k)) + 1;
%         fprintf('S(k): %d\n',S(k));
% %         [r(i,j), rb{i,j}, cE{i,j}, cands{i,j}, vt{i,j}] = nnv_net.evaluateRBN(images(k), labels(k), reachMethod, numCores, relaxFactor , disp_opt, lp_solver)
%         [r, rb, cE, cands, vt] = nnv_net.evaluateRBN(image, label, reachMethod, numCores, relaxFactor , disp_opt, lp_solver)
%     end
%     start = 1;
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %                            CIFAR10 FNNbig tanh
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % All = [3,30,51,55,67,74,93,117,131,137,155,194,226,236,245,291,298,299,311,...
% %     339,345,349,363,380,391,396,407,432,440,448,490,494,512,517,525,537,543,...
% %     545,572,575,585,610,611,613,661,663,664,668,671,681,682,694,714,722,744,...
% %     747,758,775,807,824,841,843,853,855,872,881,886,918,935,943,947,952,954,...
% %     968,978,993,995,1001,1011,1019,1030,1068,1077,1078,1079,1081,1086,1092,...
% %     1097,1103,1104,1120,1142,1143,1144,1161,1163,1201,1215,1238,1282,1296,...
% %     1302,1304,1312,1315,1317,1321,1328,1337,1341,1365,1366,1367,1370,1398,...
% %     1409,1434,1449,1450,1474,1489,1509,1520,1541,1556,1570,1580,1590,1615,...
% %     1618,1642,1651,1652,1668,1670,1680,1681,1692,1707,1715,1723,1727,1730,...
% %     1743,1744,1749,1755,1758,1777,1779,1782,1786,1796,1840,1852,1857,1865,...
% %     1867,1870,1876,1897,1913,1918,1950,1960,1974,1989,1990,2008,2013,2021,...
% %     2022,2029,2036,2042,2047,2049,2051,2053,2061,2080,2085,2090,2125,2140,...
% %     2151,2191,2201,2212,2215,2218,2226,2240,2242,2266,2267,2268,2275,2278];
% 
% S = []; %CIFAR10_FNNbig_tanh
% 
% 
% % IM = [IM_labels(S) IM_data(:,S)'];
% % writematrix(IM,'CIFAR10/data/CIFAR10_FNNbig_tanh.csv');
% % 
% % % for i=1:K
% % %     for j=1:M
% % %         eps(j)
% % %         images = attack_images(IM_data(:,S), eps(j), reachMethod, normalized); 
% % %         labels = IM_labels(S)+1;
% % %         t = tic;
% % %         [r(i,j), rb{i,j}, cE{i,j}, cands{i,j}, vt{i,j}] = nnv_net.evaluateRBN(images, labels, reachMethod, numCores, relaxFactor , disp_opt, lp_solver);
% % %         total_vt(i,j) = toc(t);
% % %     end
% % % end
% % 
% disp_opt = 'display';
% numCores = 16
% start = 11
% for j=1:M
%     fprintf('\tepsilon: %f\n',eps(j)');
%     for k = start:length(S)
%         k
%         image = attack_images(IM_data(:,S(k)), eps(j), reachMethod, normalized); 
%         label = IM_labels(S(k)) + 1;
%         fprintf('S(k): %d\n',S(k));
% %         [r(i,j), rb{i,j}, cE{i,j}, cands{i,j}, vt{i,j}] = nnv_net.evaluateRBN(images(k), labels(k), reachMethod, numCores, relaxFactor , disp_opt, lp_solver)
%         [r, rb, cE, cands, vt] = nnv_net.evaluateRBN(image, label, reachMethod, numCores, relaxFactor , disp_opt, lp_solver)
%     end
%     start = 1;
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %                            CIFAR10 FNNmed sigmoid
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AbsDom = [4,8,11,30,32,34,50,75,80,93,98,113,117,127,129,149,154,155,159,164,...
%     165,167,180,203,231,233,240,256,273,275,276,298,299,310,319,328,334,335,...
%     339,345,346,362,393,396,407,414,422,423,432,448,473,475,477,490,496,497,...
%     503,512,513,515,525,532,536,537,540,543,545,554,585,599,603,611,615,641,...
%     643,668,671,682,686,700,718,730,743,747,750,775,786,815,818,829,855,858,...
%     865,869,876,884,886,889,903,904,918,926,930,938,945,947,953,960,966,976,...
%     995,1000,1001,1005,1011,1013,1030,1068,1073,1077,1078,1103,1104,1117,1118,...
%     1138,1153,1162,1179,1198,1211,1226,1236,1238,1240,1264,1274,1296,1309,1313,...
%     1338,1340,1350,1366,1389,1395,1398,1401,1404,1408,1419,1422,1464,1490,1500,...
%     1520,1533,1543,1544,1549,1556,1560,1569,1572,1597,1618,1637,1638,1647,1650,...
%     1651,1665,1666,1668,1670,1682,1684,1687,1695,1701,1715,1719,1723,1726,1727,...
%     1729,1737,1739,1740,1746,1749,1753,1775,1782,1786,1797,1819,1848,1852,1867];
% 
% S = [4,8,11,30,34,50,80,93,98,113,117,149,155,159,164,...
%     167,180,231,233,240,256,273,276,298,299,310,319,328,334,335,...
%     345,346,362,393,396,407,414,422,432,448,475,477,490,497,...
%     503,512,513,525,532,536,537,543,545,554,585,599,603,611,615,641,...
%     668,682,718,730,750,775,786,815,818,855,...
%     865,869,884,886,889,903,904,938,945,953,960,966,995,1011,1013,1030,1068,...
%     1073,1077,1078,1103,1104,1117,1118,1153,1179,1211,1226,1236,1274]; %CIFAR10_FNNmed_sigmoid
% 
% Extra = [1309,1313,...
%     1340,1350,1366,1389,1395,1398,1401,1404];
% 
% IM = [IM_labels(S) IM_data(:,S)'];
% writematrix(IM,'CIFAR10/data/CIFAR10_FNNmed_sigmoid.csv');
% 
%  
% % % for i=1:K
% % %     for j=1:M
% % %         eps(j)
% % %         images = attack_images(IM_data(:,S), eps(j), reachMethod, normalized); 
% % %         labels = IM_labels(S)+1;
% % %         t = tic;
% % %         [r(i,j), rb{i,j}, cE{i,j}, cands{i,j}, vt{i,j}] = nnv_net.evaluateRBN(images, labels, reachMethod, numCores, relaxFactor , disp_opt, lp_solver);
% % %         total_vt(i,j) = toc(t);
% % %     end
% % % end
% % 
% disp_opt = 'display';
% numCores = 8
% start = 24
% for j=7:M
%     fprintf('\tepsilon: %f\n',eps(j)');
%     for k = start:length(S)
%         k
%         image = attack_images(IM_data(:,S(k)), eps(j), reachMethod, normalized); 
%         label = IM_labels(S(k)) + 1;
%         fprintf('S(k): %d\n',S(k));
% %         [r(i,j), rb{i,j}, cE{i,j}, cands{i,j}, vt{i,j}] = nnv_net.evaluateRBN(images(k), labels(k), reachMethod, numCores, relaxFactor , disp_opt, lp_solver)
%         [r, rb, cE, cands, vt] = nnv_net.evaluateRBN(image, label, reachMethod, numCores, relaxFactor , disp_opt, lp_solver)
%     end
%     start = 1;
% end
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            CIFAR10 FNNbig tanh
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
AbsDom = [10,46,51,61,62,67,69,82,83,104,105,106,123,132,135,138,162,176,...
    194,205,209,218,232,241,242,252,281,284,291,306,309,318,324,337,352,364,...
    370,375,382,391,401,440,455,463,491,494,495,514,531,537,539,543,546,562,...
    567,569,572,573,579,605,608,609,610,611,620,623,625,646,654,657,658,660,...
    663,677,689,699,717,724,727,733,737,739,740,754,760,765,782,787,797,...
    802,831,839,841,845,866,870,872,875,888,892,895,896,907,912,916,935,942,...
    948,952,954,962,969,974,979,988,991,998,1006,1017,1021,1022,1038,1048,1049,...
    1055,1062,1069,1071,1075,1081,1089,1099,1116,1120,1130,1132,1135,1142,1157,...
    1159,1163,1169,1174,1183,1186,1188,1191,1192,1201,1207,1213,1230,1233,1235,...
    1239,1246,1259,1275,1283,1289,1302,1307,1314,1321,1328,1333,1336,1362,1364,...
    1369,1372,1373,1397,1405,1406,1409,1411,1413,1414,1415,1418,1434,1436,1438,...
    1445,1455,1458,1465,1468,1474,1476,1481,1494,1498,1501,1505,1510,1512,...
    1518,1520,1522,1546,1550,1565,1570,1605,1615,1622,1632,1636,1642,1655,...
    1678,1680,1693,1707,1712,1717,1723,1734,1744,1752,1755,1762,1770,1780,...
    1782,1785,1787,1798,1807,1811,1812,1818,1830,1840,1850,1851,1854,1865,...
    1883,1900,1905,1908,1914,1916,1924,1926,1931,1937,1938,1967,1969,1970,...
    1971,1982,1994,2001,2021,2032,2043,2046,2055,2064,2081,2085,2090,2093,...
    2100,2104,2108,2113,2144,2165,2174,2176,2177,2180,2221,2225,2226,2239,...
    2253,2259,2267,2272,2275,2280,2283,2290,2295,2307,2309,2312,2314,2328,2334,2335];

RS = [10,67,82,83,105,123,138,162,...
    194,205,209,242,284,291,309,...
    370,391,401,440,463,491,494,514,537,543,546,...
    567,569,572,605,620,623,646,657,658,...
    663,689,699,724,739,754,760,782,787,797,...
    802,841,870,872,875,895,896,942,...
    954,962,988,991,1017,1021,1049,...
    1062,1081,1089,1120,1135,1142,...
    1159,1188,1213,1233,1235,...
    1259,1283,1289,1302,1314,1321,1364,...
    1372,1373,1397,1405,1406,1409,1411,1413,1415,1438,...
    1468,1474,1501,1505,1510,1512,1518,1520,1550,1565]; %CIFAR10_FNNbig_sigmoid

S = [1570];
% IM = [IM_labels(S) IM_data(:,S)'];
% writematrix(IM,'CIFAR10/data/CIFAR10_FNNbig_tanh.csv');

 
% % for i=1:K
% %     for j=1:M
% %         eps(j)
% %         images = attack_images(IM_data(:,S), eps(j), reachMethod, normalized); 
% %         labels = IM_labels(S)+1;
% %         t = tic;
% %         [r(i,j), rb{i,j}, cE{i,j}, cands{i,j}, vt{i,j}] = nnv_net.evaluateRBN(images, labels, reachMethod, numCores, relaxFactor , disp_opt, lp_solver);
% %         total_vt(i,j) = toc(t);
% %     end
% % end
% 
disp_opt = 'display';
numCores = 1
start = 1
for j=1:M
    fprintf('\tepsilon: %f\n',eps(j)');
    for k = start:length(S)
        k
        image = attack_images(IM_data(:,S(k)), eps(j), reachMethod, normalized); 
        label = IM_labels(S(k)) + 1;
        fprintf('S(k): %d\n',S(k));
%         [r(i,j), rb{i,j}, cE{i,j}, cands{i,j}, vt{i,j}] = nnv_net.evaluateRBN(images(k), labels(k), reachMethod, numCores, relaxFactor , disp_opt, lp_solver)
        [r, rb, cE, cands, vt] = nnv_net.evaluateRBN(image, label, reachMethod, numCores, relaxFactor , disp_opt, lp_solver)
    end
    start = 1;
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% RS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,...
%     26,27,28,29,30,31,32,33,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51,....
%     53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,....
%     78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102];



% IM = [IM_labels(S) IM_data(:,S)'];
% writematrix(IM,'MNIST/data/sigmoid_100_50_eps020.csv');

% j = 1;
% eps(j)
% images = attack_images(IM_data, eps(j), reachMethod, normalized); 
% labels = IM_labels+1;
% for s = 1:length(S)
%     [r, rb, cE, cands, vt] = nnv_net.evaluateRBN(images(S(s)), labels(S(s)), reachMethod, numCores, relaxFactor , disp_opt, lp_solver);
%     safe = [sum(rb==1)];
%     unsafe = [sum(rb == 0)];
%     unknown = [sum(rb == 2)];
%     
% %     if  rb==1 && j == 1
% %         S = [S s];
% %     end
%     if rb ~= 1
%         s;
%         fprintf('S(s): %d',S(s));
%         Ta = table(eps(j), safe, unsafe, unknown, vt);
%     end
% end
% S




% % S = [3,6,12,30,55,93,100,108,117,137,167,176,190,199,203,218,290,297,...
% %      298,332,339,340,365,380,393,407,416,448,473,478,487,490,...
% %      496,517,520,537,543,551,579,603,648,658,662,663,664,...
% %      668,671,682,694,697,701,712,714,722,744,751,757,760,775,786,795,804,...
% %      816,854,865,880,881,886,889,921,922,935,947,952,954,960,968,...
% %      986,993,...
% %      995,1004,1011,1030,1068,1074,1077,1078,1081,1091,1093,1097,1104,1105,...
% %      1112,1115,1119,1120,1126,1136,1137
% %      ]; CIFAR10_FNNsmall_tanh_0.01
% % %         ,995,1004,1011,1030,1068,1074,1077,1078,1081,1091,1093,1097,1104,1105,...
% % %      1112,1115,1119,1120,1126,1136,1137,1112,1115,1119,1120,1126,1136,1137,1140,...
% % %      1143,1144,1155,1159,1161,1163,1168,1172,1196,1202,1233,1238,1266,1273,1274,...
% % %      1283,1296,1304,1312,1315,1318];
% % 
% % IM = [IM_labels(S) IM_data(:,S)'];
% % writematrix(IM,'CIFAR10/data/CIFAR10_FNNsmall_tanh.csv');
% % 
% % % disp_opt = 'display';
% % numCores = 16
% % start = 45
% % for j=9:M
% %     fprintf('\tepsilon: %f\n',eps(j)');
% %     for k = start:length(S)
% %         k
% %         image = attack_images(IM_data(:,S(k)), eps(j), reachMethod, normalized); 
% %         label = IM_labels(S(k));
% %         fprintf('S(k): %d\n',S(k));
% % %         [r(i,j), rb{i,j}, cE{i,j}, cands{i,j}, vt{i,j}] = nnv_net.evaluateRBN(images(k), labels(k), reachMethod, numCores, relaxFactor , disp_opt, lp_solver)
% %         [r, rb, cE, cands, vt] = nnv_net.evaluateRBN(image, label, reachMethod, numCores, relaxFactor , disp_opt, lp_solver)
% %     end
% %     start = 1;
% % end


% disp_opt = 'display';
% for j=1:M
%     eps(j)
%     images = attack_images(IM_data(:,S), eps(j), reachMethod, normalized); 
%     labels = IM_labels(S);
%     for k = 1:length(S)
%         k
%         fprintf('S(k): %d\n',S(k));
% %         [r(i,j), rb{i,j}, cE{i,j}, cands{i,j}, vt{i,j}] = nnv_net.evaluateRBN(images(k), labels(k), reachMethod, numCores, relaxFactor , disp_opt, lp_solver)
%         [r, rb, cE, cands, vt] = nnv_net.evaluateRBN(images(k), labels(k), reachMethod, numCores, relaxFactor , disp_opt, lp_solver)
%     end
% end

% for i=1:length(S)
%     for j=1:M
%         images = attack_images(IM_data(S(i)), eps(j), reachMethod, normalized); 
%         labels = IM_labels(S(i))+1;
%         t = tic;
%         [r(i,j), rb{i,j}, cE{i,j}, cands{i,j}, vt{i,j}] = nnv_net.evaluateRBN(images(S(i)), labels(S(i)), reachMethod, numCores, relaxFactor , disp_opt, lp_solver);
%         total_vt(i,j) = toc(t);
%     end
% end
% 
% N = 50;



% IM = [IM_labels(S) IM_data(:,S)'];
% writematrix(IM,'tan_100_50_020.csv');





% for k = 1:length(S)
%     k
%     fprintf('S(k): %d\n',S(k));
%         for j=1:M
%             eps(j)
%             images = attack_images(IM_data(:,S(k)), eps(j), reachMethod, normalized); 
%             labels = IM_labels(S(k))+1;
%             t = tic;
%             [r(i,j), rb{i,j}, cE{i,j}, cands{i,j}, vt{i,j}] = nnv_net.evaluateRBN(images, labels, reachMethod, numCores, relaxFactor , disp_opt, lp_solver)
%             total_vt(i,j) = toc(t);
%         end
%     
% end


    



T = table;
%rf = [];
ep = [];
VT = [];
RB = [];
US = [];
UK = [];
for i=1:K
    %rf = [rf; relaxFactor(i)*ones(M,1)];
    ep = [ep; eps'];
    unsafe = zeros(M,1);
    robust = zeros(M,1);
    unknown = zeros(M,1);
    for j=1:M
        unsafe(j) = sum(rb{i,j}==0);
        robust(j) = sum(rb{i,j} == 1);
        unknown(j) = sum(rb{i,j}==2);
    end
    RB = [RB; robust];
    US = [US; unsafe];
    UK = [UK; unknown];
    VT = [VT; total_vt(i,:)'];
end
%T.relaxFactor = rf;
T.epsilon = ep;
T.robustness = RB;
T.unsafe = US;
T.unknown = UK;
T.verifyTime = VT;

fprintf('%s', reachMethod);
T

% save_ = sprintf('result/%s_%s%s_%s_%s', dataset_, net_, norm_, reachMethod, datetime('today'))
% save(save_, 'lp_solver', 'T', 'r', 'rb', 'cE', 'cands', 'vt', 'total_vt');

function images = attack_images(in_images, epsilon, reachMethod, normalized)
    if normalized
        max_px = 1.0;
    else
        max_px = 255.0;
    end
    
    
    N = size(in_images, 2);
    for n = 1:N
        image = in_images(:, n);
        if normalized
            image = image/255.0;
        end
        lb = image - epsilon;
        ub = image + epsilon;
        ub(ub > max_px) = max_px;
        lb(lb < 0.0) = 0.0;
        
        if strcmp(reachMethod,'approx-star') || strcmp(reachMethod, 'abs-dom')
            images(n) = Star(lb, ub);
        elseif strcmp(reachMethod,'rstar-absdom-two')
            images(n) = RStar(lb, ub, inf);
        elseif strcmp(reachMethod,'absdom')
            images(n) = AbsDom(lb, ub, inf);
        elseif strcmp(reachMethod,'approx-zono')
            B = Box(lb, ub);
            images(n) = B.toZono;
        else
           error('unsupported reachMethod for evaluateRBN')
        end
    end
end

function nnv_net = net2nnv_net(net)
    if strcmp(net.Layers(4).Type,'Sigmoid')
    act_fn = 'logsig';
    elseif strcmp(net.Layers(4).Type,'Tanh')
        act_fn = 'tansig';
    end

    L = [];
    for i = 3:2:length(net.Layers)-4
        L1 = LayerS(net.Layers(i).Weights, net.Layers(i).Bias, act_fn);
        L = [L L1]; 
    end
    L2 = LayerS(net.Layers(i+2).Weights, net.Layers(i+2).Bias, 'purelin');
    nnv_net = FFNNS([L L2]);
    nnv_net.lp_solver = 'linprog';
end



% classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
% net = importONNXNetwork('train/FNNbig/ffnnTANH__Point_6_500.onnx','OutputLayerType','classification','Classes',classes);
% act_fn = 'tansig';
% L1 = LayerS(reshape(net.Layers(6).Weights, [784,500])', reshape(net.Layers(6).Bias, [1 500])', act_fn);
% L2 = LayerS(reshape(net.Layers(9).Weights, [500,500])', reshape(net.Layers(9).Bias, [1 500])', act_fn);
% L3 = LayerS(reshape(net.Layers(12).Weights, [500,500])', reshape(net.Layers(12).Bias, [1 500])', act_fn);
% L4 = LayerS(reshape(net.Layers(15).Weights, [500,500])', reshape(net.Layers(15).Bias, [1 500])', act_fn);
% L5 = LayerS(reshape(net.Layers(18).Weights, [500,500])', reshape(net.Layers(18).Bias, [1 500])', act_fn);
% L6 = LayerS(reshape(net.Layers(21).Weights, [500,500])', reshape(net.Layers(21).Bias, [1 500])', act_fn);
% L7 = LayerS(reshape(net.Layers(24).Weights, [500,10])', reshape(net.Layers(24).Bias, [1 10])', act_fn);
% nnv_net = FFNNS([L1 L2 L3 L4 L5 L6 L7]);
