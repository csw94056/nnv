close all;
clear;
clc;
format long;


%%
% network trained with images: [0 1] -> normalized, 
%                              [0 255] ->  not_normalized
dataset_ = 'MNIST';
net_ = 'MNIST_FNNsmall_tanh';
normalized = 0;

norm_ = '';
if normalized
    norm_ = '_normalized'
end

net_dir = sprintf('%s.mat', net_)
image_dir = sprintf('%s.csv', net_)

normalize = 1;
%% load network
load(net_dir);
nnv_net_linprog = net2nnv_net(net, 'linprog');
nnv_net_glpk = net2nnv_net(net, 'glpk');
nnv_net_estimate = net2nnv_net(net, 'estimate');

%% load images
csv_data = csvread(image_dir);
IM_labels = csv_data(:,1);
IM_data = csv_data(:,2:end)';

numCores = 1;
disp_opt = 0;

% eps = [0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02];
eps = 0.01;

N = size(IM_data, 2);
M = length(eps);

l = length(nnv_net_glpk.Layers);
n = 5 %26
% j = 1

% figure                                          % initialize figure
% colormap(gray)                                  % set to grayscale
% digit = reshape(IM_data(:, n), [28,28]);     % row = 28 x 28 image
% imagesc(digit)                              % show the image
% title(num2str(IM_labels(n)))                   % show the label

for j = 1:M
    fprintf('epsilon: %f\n\n', eps(j));
    for n = 5:N
        fprintf('image: %d\n\n', n);
        S_linprog = attack_images(IM_data(:,n), eps(j), 'approx-star', normalize);
%         S_linprog = attack_images(IM_data(:,n), eps(j), 'abs-dom', normalize);
        S_glpk = S_linprog;
        S_estimate = S_linprog;
        RS = attack_images(IM_data(:,n), eps(j), 'rstar-absdom-two', normalize); 
        
        labels = IM_labels(n)+1;
        for i = 1:l
            % Affine Mapping
            fprintf('layers: %d\n', i);
            fprintf('Affine Mapping\n');

            S_glpk = S_glpk.affineMap(nnv_net_glpk.Layers(i).W, nnv_net_glpk.Layers(i).b);
            S_linprog = S_linprog.affineMap(nnv_net_linprog.Layers(i).W, nnv_net_linprog.Layers(i).b);
            S_estimate = S_estimate.affineMap(nnv_net_estimate.Layers(i).W, nnv_net_estimate.Layers(i).b);
            RS = RS.affineMap(nnv_net_linprog.Layers(i).W, nnv_net_linprog.Layers(i).b);
            
            % Check infeasibility of set after affine mapping
            S_glpk_empty = S_glpk.isEmptySet;
            S_linprog_empty = S_linprog.isEmptySet;
            S_estimate_empty = S_estimate.isEmptySet;
            RS_empty = RS.isEmptySet;
            
            if S_glpk_empty
                fprintf('S_glpk is an infeasible set\n');
            end
            if S_linprog_empty
                fprintf('S_linprog is an infeasible set\n');
            end
            if S_estimate_empty
                fprintf('S_estimate is an infeasible set\n');
            end
            if RS_empty
                fprintf('RS is an infeasible set\n');
            end

            % Get predicate bounds of each sets
            dim = S_glpk.dim;
            l_glpk = zeros(dim,1); u_glpk = zeros(dim,1);
            l_linprog = zeros(dim,1); u_linprog = zeros(dim,1);
            for d = 1:dim
%                 fprintf('neuron: %d\n', d);
                l_glpk(d) = S_glpk.getMin(d,'glpk');
                u_glpk(d) = S_glpk.getMax(d,'glpk');

                l_linprog(d) = S_linprog.getMin(d,'linprog');
                u_linprog(d) = S_linprog.getMax(d,'linprog');
            end
            [l_estimate, u_estimate] = S_estimate.estimateRanges;
            [l_rs, u_rs] = RS.getRanges;
            
            if i == l
                break;
            end
        
            % Activation Function
            fprintf('Activation Function\n');
            S_glpk = TanSig.multiStepTanSig_NoSplit(S_glpk, 0, 'glpk');
            S_linprog = TanSig.multiStepTanSig_NoSplit(S_linprog, 0, 'linprog');
%             S_estimate = TanSig.multiStepTanSig_NoSplit(S_estimate
%             S_glpk = TanSig.reach_abstract_domain_approx(S_glpk, 'glpk');
%             S_linprog = TanSig.reach_abstract_domain_approx(S_linprog, 'linprog');
%             S_estimate = TanSig.reach_abstract_domain_approx(S_estimate, 'estimate');
%             RS = TanSig.reach_rstar_absdom_with_two_pred_const(RS);
            
            % Check infeasibility of set after affine mapping

            S_glpk_empty = S_glpk.isEmptySet;
            S_linprog_empty = S_linprog.isEmptySet;
%             S_estimate_empty = S_estimate.isEmptySet;
            RS_empty = RS.isEmptySet;
            
            if S_glpk_empty
                fprintf('S_glpk is an infeasible set\n');
            end
            if S_linprog_empty
                fprintf('S_linprog is an infeasible set\n');
            end
%             if S_estimate_empty
%                 fprintf('S_estimate is an infeasible set\n');
%             end
            if RS_empty
                fprintf('RS is an infeasible set\n');
            end
            
            % Get predicate bounds of each sets
            dim = S_glpk.dim;
            l_glpk = zeros(dim,1); u_glpk = zeros(dim,1);
            l_linprog = zeros(dim,1); u_linprog = zeros(dim,1);
            for d = 1:dim
                l_glpk(d) = S_glpk.getMin(d,'glpk');
                u_glpk(d) = S_glpk.getMax(d,'glpk');

                l_linprog(d) = S_linprog.getMin(d,'linprog');
                u_linprog(d) = S_linprog.getMax(d,'linprog');
            end
%             [l_estimate, u_estimate] = S_estimate.estimateRanges;
            [l_rs, u_rs] = RS.getRanges;
        end
    end
end
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