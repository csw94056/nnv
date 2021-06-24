close all;
clear;
clc;
format long;

%%
% network trained with images: [0 1] -> normalized, 
%                              [0 255] ->  not_normalized
dataset_ = 'MNIST';
% net_ = 'sigmoid_100_50';
net_ = 'MNIST_FNNmed_tanh';
n_ = 'FNNmed';
normalized = 0;


norm_ = '';
if normalized
    norm_ = '_normalized'
end

net_dir = sprintf('%s/nets/%s/%s.mat', dataset_,n_,net_)
% image_dir = sprintf('%s/data/%s.csv', dataset_,net_)
image_dir = sprintf('%s/data/%s_raw.csv', dataset_,net_)


normalized = 1;
%% load network
load(net_dir);
nnv_net = net2nnv_net(net);

%% load images
csv_data = csvread(image_dir);
IM_labels = csv_data(:,1);
IM_data = csv_data(:,2:end)';

% IM = [IM_labels(81) IM_data(:,81)'];
% writematrix(IM,'image3.csv');

relaxFactor = [0];
numCores = 1;
disp_opt = 0;
lp_solver = 'glpk' % 'linprog''

% eps = [0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02];
eps = 0.01;

N = size(IM_data, 2);
K = length(relaxFactor);
M = length(eps);

r = zeros(K, M); % percentage of images that are robust
rb = cell(K, M); % detail robustness verification result
cE = cell(K, M); % detail counterexamples
vt = cell(K, M); % detail verification time
cands = cell(K,M); % counterexample
total_vt = zeros(K, M); % total verification time

l = length(nnv_net.Layers);
n = 1 %26
j = 1

figure                                          % initialize figure
colormap(gray)                                  % set to grayscale
digit = reshape(IM_data(:,n), [28,28]);     % row = 28 x 28 image
imagesc(digit)                              % show the image
title(num2str(IM_labels(n)))                   % show the label


% reachMethod = 'approx-star';
% reachMethod = 'abs-dom';    %Star abstract domain (LP)
% reachMethod = 'rstar-absdom-two';
% reachMethod = 'absdom';
% reachMethod = 'approx-zono';

RS = attack_images(IM_data(:,n), eps(j), 'rstar-absdom-two', normalized); 
labels = IM_labels(n)+1;
A = attack_images(IM_data(:,n), eps(j), 'abs-dom', normalized);
S = attack_images(IM_data(:,n), eps(j), 'approx-star', normalized);
Z = attack_images(IM_data(:,n), eps(j), 'approx-zono', normalized);

for i = 1:l
    % Affine Mapping
    fprintf('index: %d\n', i);
    fprintf('Affine Mapping\n');

    RS = RS.affineMap(nnv_net.Layers(i).W, nnv_net.Layers(i).b);
    A = A.affineMap(nnv_net.Layers(i).W, nnv_net.Layers(i).b);
    S = S.affineMap(nnv_net.Layers(i).W, nnv_net.Layers(i).b);
    Z = Z.affineMap(nnv_net.Layers(i).W, nnv_net.Layers(i).b);
    
    % Check infeasibility of set after affine mapping
    RS_empty = RS.isEmptySet;
    A_empty = A.isEmptySet;
    S_empty = S.isEmptySet;
    Z_star = Z.toStar;
    Z_empty  = Z_star.isEmptySet;
    
    if RS_empty
        fprintf('RS is an infeasible set\n');
    end
    if A_empty
        fprintf('A is an infeasible set\n');
    end
    if S_empty
        fprintf('S is an infeasible set\n');
    end
    if Z_empty
        fprintf('Z is an infeasible set\n');
    end
    
    % Get predicate bounds of each sets
    [l_rs, u_rs] = RS.getRanges;
    [l_a, u_a] = A.getRanges;
    [l_s, u_s] = S.getRanges;
    [l_z, u_z] = Z.getRanges;
    
    
    % Activation Function
    fprintf('\nActivation Function\n');
    RS = TanSig.reach_rstar_absdom_with_two_pred_const(RS);
    A = TanSig.reach_abstract_domain_approx(A);
    S = TanSig.multiStepTanSig_NoSplit(S);
    Z = TanSig.reach_zono_approx(Z);
    
    % Check infeasibility of set after affine mapping
    RS_empty = RS.isEmptySet;
    A_empty = A.isEmptySet;
    S_empty = S.isEmptySet;
    Z_star = Z.toStar;
    Z_empty  = Z_star.isEmptySet;
    
    if RS_empty
        fprintf('RS is an infeasible set\n');
    end
    if A_empty
        fprintf('A is an infeasible set\n');
    end
    if S_empty
        fprintf('S is an infeasible set\n');
    end
    if Z_empty
        fprintf('Z is an infeasible set\n');
    end
    
    % Get predicate bounds of each sets
    [l_rs, u_rs] = RS.getRanges;
    [l_a, u_a] = A.getRanges;
    [l_s, u_s] = S.getRanges;
    [l_z, u_z] = Z.getRanges;
end

% for i = 1:l
%     fprintf('index: %d\n', i);
%     fprintf('Affine Mapping\n');
%     RS = RS.affineMap(nnv_net.Layers(i).W, nnv_net.Layers(i).b);
% %     [l_rs, u_rs] = printf_bounds(RS);
% %     fprintf('\nupper and lower bounds difference');
% %     bounds_diff = u_rs - l_rs
%     A = A.affineMap(nnv_net.Layers(i).W, nnv_net.Layers(i).b);
%     S = S.affineMap(nnv_net.Layers(i).W, nnv_net.Layers(i).b);
%     Z = Z.affineMap(nnv_net.Layers(i).W, nnv_net.Layers(i).b);
%     
% % %     Z = Z.affineMap(nnv_net.Layers(i).W, nnv_net.Layers(i).b);
% % %     [l_z, u_z] = printf_bounds(Z);
% % %     fprintf('upper and lower bounds difference');
% % %     bounds_diff = u_z - l_z
% % %     
% % %     fprintf('\nbounds difference between two sets');
% % %     l_diff = l_z - l_rs
% % %     u_diff = u_z - l_rs
%     
%     fprintf('\nActivation Function\n');
%     if isa(RS,'RStar')
%         RS = TanSig.reach_rstar_absdom_with_two_pred_const(RS);
%     elseif isa(RS, 'Star')
%         if strcmp(reachMethod, 'abs-dom')
%             RS = TanSig.reach_abstract_domain_approx(RS);
%         elseif strcmp(reachMethod, 'approx-star')
%             RS = TanSig.multiStepTanSig_NoSplit(RS);
%         end
%     end
%     [l_rs, u_rs] = printf_bounds(RS);
% %     fprintf('upper and lower bounds difference');
% %     bounds_diff = u_rs - l_rs
%     
%     S = TanSig.reach_abstract_domain_approx(S);
%     [l_s, u_s] = printf_bounds(S);
% %     fprintf('upper and lower bounds difference');
% %     bounds_diff = u_s - l_s
% 
%     fprintf('bounds difference between Star and RStar');
%     l_diff = l_rs - l_s
%     u_diff = u_rs - l_s
%     
% % %     Z = TanSig.reach_zono_approx(Z);
% % %     [l_z, u_z] = printf_bounds(Z);
% % %     fprintf('\nupper and lower bounds difference');
% % %     bounds_diff = u_z - l_z
% % %     
% % %     fprintf('\nbounds difference between two sets');
% % %     l_diff = l_z - l_rs
% % %     u_diff = u_z - l_rs
% end

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
    lp_solver_ = 'linprog';
    
    if strcmp(net.Layers(4).Type,'Sigmoid')
    act_fn = 'logsig';
    elseif strcmp(net.Layers(4).Type,'Tanh')
        act_fn = 'tansig';
    end

    L = [];
    for i = 3:2:length(net.Layers)-4
        L1 = LayerS(net.Layers(i).Weights, net.Layers(i).Bias, act_fn);
        L1.lp_solver = lp_solver_;
        L = [L L1]; 
    end
    L2 = LayerS(net.Layers(i+2).Weights, net.Layers(i+2).Bias, 'purelin');
    L2.lp_solver = lp_solver_;
    nnv_net = FFNNS([L L2]);
    nnv_net.lp_solver = lp_solver_;
end


function [l, u] = printf_bounds(RS)
    if isa(RS, 'Zono')
        S = RS.toStar;
        a = S.isEmptySet;
    elseif isa(RS, 'RStar')
        a = 0;
    else
        a = RS.isEmptySet;
    end
    
    
    if a
        error('RS is an infeasible set!!!!!!!!!!!!!!!!!!!!!!!!!!!');
    end
    if isa(RS, 'RStar')
        [l, u] = RS.getRanges;
        [ls, us] = RS.getExactRanges;
        fprintf('\nabsdom\t\t\texact (lower bound) of RStar\n');
        for j = 1:size(l)
            fprintf('%1.27f\t\t%1.27f\n',l(j),ls(j));
        end
        fprintf('\nlower bound difference\n');
        fprintf('%1.27f\n', l - ls);

        fprintf('\nabsdom\t\t\texact (upper bound)\n');
        for j = 1:size(l)
            fprintf('%1.27f\t\t%1.27f\n',u(j),us(j));
        end
        fprintf('\nupper bound difference\n');
        fprintf('%1.27f\n', u - us);
        
    elseif isa(RS, 'Star') | isa(RS, 'AbsDom') | isa(RS, 'Zono')
        if isa(RS, 'Star')
            fprintf('Star set');
        elseif isa(RS, 'AbsDom')
            fprintf('AbsDom set');
        elseif isa(RS, 'Zono');
            fprintf('Zono set');
        end
        [l, u] = RS.getRanges;
        fprintf('\nlower bounds\n');
        fprintf('%1.27f\n', l);
        
        fprintf('\nupper bounds\n');
        fprintf('%1.27f\n', u);
    end
end