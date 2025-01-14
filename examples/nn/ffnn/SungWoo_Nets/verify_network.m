close all;
clear;
clc;
format long;

%%
% network trained with images: [0 1] -> normalized, 
%                              [0 255] ->  not_normalized
dataset_ = 'MNIST';
net_ = 'MNIST_FNNbig_tanh';
n_ = 'FNNbig';
normalized = 0;


norm_ = '';
if normalized
    norm_ = '_normalized'
end

net_dir = sprintf('%s/nets/%s/%s.mat', dataset_,n_,net_)
% net_dir = sprintf('%s/nets/%s/MNIST_%s%s_DenseNet.mat', dataset_,net_, net_, norm_);
% image_dir = sprintf('MNIST/data/%s%s.csv',net_,norm_);
% image_dir = sprintf('MNIST/data/%s_raw.csv',net_);
% image_dir = sprintf('MNIST/data/%s%s_eps020.csv', net_, norm_)

image_dir = sprintf('%s/data/%s.csv', dataset_,net_)
% image_dir = sprintf('%s/data/%s_raw.csv', dataset_,net_)
% image_dir = sprintf('%s/data/%s_zono.csv', dataset_,net_)


% reachMethod = 'approx-star'
% reachMethod = 'abs-dom'    %Star abstract domain (LP)
reachMethod = 'rstar-absdom-two'
% reachMethod = 'absdom'
% reachMethod = 'approx-zono'

relaxFactor = [0];
numCores = 1;
disp_opt = 0; %'display';
lp_solver = 'linprog' % 'linprog'
normalized = 1;
%% load network
load(net_dir);
nnv_net = net2nnv_net(net, lp_solver);

%% load images
csv_data = csvread(image_dir);
IM_labels = csv_data(:,1);
IM_data = csv_data(:,2:end)';

%eps = [0, 1, 2, 3, 4, 5, 6, 7, 8];
% eps = [0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.024, 0.026, 0.028, 0.030];

eps = [0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02];

% eps = [0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.002];
% eps = [0.011, 0.013, 0.015, 0.017, 0.019];
% eps = 0.01;
% eps = 0.0000000000000001;
% eps = [0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012,0.013,0.014,0.015];
% eps = [0.005, 0.006];
% eps = 0.005

N = size(IM_data, 2);
K = length(relaxFactor);
M = length(eps);

r = zeros(K, M); % percentage of images that are robust
rb = cell(K, M); % detail robustness verification result
cE = cell(K, M); % detail counterexamples
vt = cell(K, M); % detail verification time
cands = cell(K,M); % counterexample
total_vt = zeros(K, M); % total verification time


% S = [];
% j = 1;
% eps(j)
% images = attack_images(IM_data, eps(j), reachMethod, normalized); 
% labels = IM_labels+1;
% for s = 1:N
%     [r, rb, cE, cands, vt] = nnv_net.evaluateRBN(images(s), labels(s), reachMethod, numCores, relaxFactor , disp_opt, lp_solver);
%     if  rb==1 && j == 1
%         S = [S s];
%     end
% end
% 
% 
% IM = [IM_labels(S) IM_data(:,S)'];
% writematrix(IM,'sigmoid_100_100.csv');


% R = cell(N, M);

% for j = 1:M
%     R = [];
%     E = zeros(1,N);
%     fprintf('\tepsilon: %f\n', eps(j));
%     for n = 1:N
%         fprintf('%d\n',n);
%         images = attack_images(IM_data(:,n), eps(j), reachMethod, normalized); 
%         labels = IM_labels(n)+1;
% 
% %         [R{j,n} t] = nnv_net.reach(images, reachMethod, numCores, 0, 0, 'linprog');
%         [S, t] = nnv_net.reach(images, reachMethod, numCores, 0, 0, 'linprog');
%         R = [R S];
%         
%         E(n) = isEmptySet(R(n));
%         if(E(n))
%             fprintf('%d is an empty set!!!!!!!!!!!!!!!!!!!!!!!\n', n);
%         else
%             fprintf('%d is NOT an empty set\n', n);
%         end
%     end
% end
%     
%     fprintf('isEmptySet?\n');
%     for n = 1:N
%        fprintf('%d\n',n);
%        E(n) = isEmptySet(R(n));
%     end
%     find(E == 1)
% end
% 
% for j = 1:1
%     for n = 1:N
%        fprintf('%d\n',n);
%        E(n) = isEmptySet(R(n));
%     end
% end

for i=1:K
    for j=1:1
        eps(j)
        images = attack_images(IM_data, eps(j), reachMethod, normalized); 
        labels = IM_labels+1;
        t = tic;
        [r(i,j), rb{i,j}, cE{i,j}, cands{i,j}, vt{i,j}] = nnv_net.evaluateRBN(images(1:N), labels(1:N), reachMethod, numCores, relaxFactor , disp_opt, lp_solver);
        total_vt(i,j) = toc(t);
    end
end


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

save_ = sprintf('result/%s_%s%s_%s_%s', dataset_, net_, norm_, reachMethod, datetime('today'))
save(save_, 'lp_solver', 'T', 'r', 'rb', 'cE', 'cands', 'vt', 'total_vt');

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

function nnv_net = net2nnv_net(net, lp_solver)  
    if strcmp(net.Layers(4).Type,'Sigmoid')
    act_fn = 'logsig';
    elseif strcmp(net.Layers(4).Type,'Tanh')
        act_fn = 'tansig';
    end

    L = [];
    for i = 3:2:length(net.Layers)-4
        L1 = LayerS(net.Layers(i).Weights, net.Layers(i).Bias, act_fn);
        L1.lp_solver = lp_solver;
        L = [L L1]; 
    end
    L2 = LayerS(net.Layers(i+2).Weights, net.Layers(i+2).Bias, 'purelin');
    L2.lp_solver = lp_solver;
    nnv_net = FFNNS([L L2]);
    nnv_net.lp_solver = lp_solver;
end
