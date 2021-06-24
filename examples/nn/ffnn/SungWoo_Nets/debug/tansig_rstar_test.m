close all;
clear;
clc;
format long;

input_dim = 2;
output_dim = 2;
% n is number of neurons per hidden layers
n = 2;
% h is number of hidden layers
h = 2;

display = 1;
% if input_dim <= 3
%     display = 1;
% end

% W{1} = -1 + 2*rand(n,input_dim);
% b{1} = -1 + 2*rand(n, 1);
% for i = 2:h
%     W{i} = -1 + 2*rand(n,n);
%     b{i} = -1 + 2*rand(n,1);
% end
% W{i+1} = -1 + 2*rand(output_dim, n);
% b{i+1} = -1 + 2*rand(output_dim, 1);

W{1} = -0.1 + 0.2*rand(n,input_dim);
b{1} = -0.1 + 0.2*rand(n, 1);
for i = 2:h
    W{i} = -0.1 + 0.2*rand(n,n);
    b{i} = -0.1 + 0.2*rand(n,1);
end
W{i+1} = -0.1 + 0.2*rand(output_dim, n);
b{i+1} = -0.1 + 0.2*rand(output_dim, 1);

% save Wb_tansig_test.mat W b;
% load Wb_tansig.mat
% load Wb_tansig_test.mat;
steps = length(W);
% steps = 5;

% lb = -ones(input_dim, 1);
lb = zeros(input_dim, 1);
ub = ones(input_dim, 1);

% p{1} = (lb + ub) * 0.5;
% p{2} = lb;
% p{3} = ub;
% p{4} = [ub(1); lb(1)];
% p{5} = [lb(1); ub(2)];

k = 1;
% for n = lb(1):ub(1)/5.0:(ub(1)-lb(1))/2.0
%     for m = lb(2):ub(2)/5.0:(ub(2)-lb(2))/2.0
%         p{k} = [n; m];
%         k = k+1;
%     end
% end


for n = lb(1):ub(1)/10.0:(ub(1)-lb(1))
    for m = lb(2):ub(2)/10.0:(ub(2)-lb(2))
        p{k} = [n; m];
        k = k+1;
    end
end


P = Polyhedron('lb', lb, 'ub', ub);
A = AbsDom(P, inf);
R2 = RStar(P,inf);
S = Star(P);
S.predicate_lb = -ones(input_dim, 1);
S.predicate_ub = ones(input_dim, 1);
Sa = S;
B = Box(lb,ub);
Z = B.toZono;

if display
    figure;
    nexttile;
    plot(A, 'r');
    hold on;
    plot(Z, 'g');
    hold on;
    plot(Sa, 'b');
    hold on;
    plot(R2, 'y');
    hold on
    plot(S, 'c');          % Star approx
    hold on;
    for i = 1:length(p)
        plot(p{i}(1),p{i}(2), '*', 'color', 'black');
    end
    title('input');
end

% %legend('AbsDom','Star approx', 'RStar 2');
legend('AbsDom', 'Zono', 'Star AbsDom', 'RStar', 'Star approx');

for i = 1:steps
    A = A.affineMap(W{i}, b{i});
    Sa = Sa.affineMap(W{i}, b{i});
    R2 = R2.affineMap(W{i}, b{i});
    S = S.affineMap(W{i}, b{i});
    Z = Z.affineMap(W{i}, b{i});
    if display
        nexttile;
        plot(A, 'r');
        hold on;
        plot(Z, 'g');
        hold on;
        plot(Sa, 'b');
        hold on;
        plot(R2, 'y');
        hold on;
        plot(S, 'c');
        hold on;
        for j = 1:length(p)
            p{j} = point_affineMap(p{j}, W{i}, b{i});
            if length(p{j}) == 3
                plot3(p{j}(1),p{j}(2),p{j}(3), '*', 'color', 'black');
                xlabel('x1');
                ylabel('x2');
                zlabel('x3');
            else
                plot(p{j}(1),p{j}(2), '*', 'color', 'black');
                xlabel('x1');
                ylabel('x2');
            end
        end
    
    %     hold off;
        title(sprintf('Affine Map (%d)', i));
    end
    fprintf('\nAffine Map (%d)', i);
    fprintf('\nAbsdom bounds');
    [l, u] = A.getRanges
    fprintf('Zono bounds');
    [l, u] = Z.getRanges
    fprintf('RStar Exact bounds');
    [l, u] = R2.getExactRanges
    fprintf('RStar approx bounds');
    [l, u] = R2.getRanges
    fprintf('Star absdom approx bounds');
    [l, u] = Sa.getRanges
    fprintf('Star approx bounds');
    [l, u] = S.getRanges
    
    if i == steps
        break;
    end
    
    A = TanSig.reach_absdom_approx(A);
    Sa = TanSig.reach_abstract_domain_approx(Sa);
    R2 = TanSig.reach_rstar_absdom_with_two_pred_const(R2);
    S = TanSig.multiStepTanSig_NoSplit(S);
    Z = TanSig.reach_zono_approx(Z);
    if display
        nexttile;
        plot(A, 'r');
        hold on;
        plot(Z, 'g');
        hold on;
        plot(Sa, 'b');
        hold on;
        plot(R2, 'y');
        hold on;
        plot(S, 'c');
        hold on;
        for j = 1:length(p)
            p{j} = point_reach(p{j});
            if length(p{j}) == 3
                plot3(p{j}(1),p{j}(2),p{j}(3), '*', 'color', 'black');
                xlabel('x1');
                ylabel('x2');
                zlabel('x3');
            else
                plot(p{j}(1),p{j}(2), '*', 'color', 'black');
                xlabel('x1');
                ylabel('x2');
            end
        end
    %     hold off;
        title(sprintf('Activation Function (%d)', i));
    end
    fprintf('\nActivation Function (%d)', i);
    fprintf('\nAbsdom bounds');
    [l, u] = A.getRanges
    fprintf('Zono bounds');
    [l, u] = Z.getRanges
    fprintf('RStar Exact bounds');
    [l, u] = R2.getExactRanges
    fprintf('RStar approx bounds');
    [l, u] = R2.getRanges
    fprintf('Star absdom approx bounds');
    [l, u] = Sa.getRanges
    fprintf('Star approx bounds');
    [l, u] = S.getRanges
end
if display
    figure('Name','Reachability Result');
    nexttile;
    plot(A, 'r');
    hold on;
    title('AbsDom');
    nexttile;
    plot(Z, 'g');
    hold on;
    title('Zono');
    % nexttile;
    % plot(Sa, 'b');
    % title('Star AbsDom');
    nexttile;
    plot(R2, 'y');
    title('RStar 2');
    nexttile;
    plot(S, 'c');
    title('Star approx');
    nexttile;
    hold on;
    for j = 1:length(p)
        if length(p{j}) == 3
            plot3(p{j}(1),p{j}(2),p{j}(3), '*', 'color', 'black');
            xlabel('x1');
            ylabel('x2');
            zlabel('x3');
        else
            plot(p{j}(1),p{j}(2), '*', 'color', 'black');
            xlabel('x1');
            ylabel('x2');
        end
    end
    % hold off;
    title('dot plots');

    nexttile;
    plot(A, 'r');
    hold on;
    plot(Z, 'g');
    hold on;
    plot(Sa, 'b');
    hold on;
    plot(R2, 'y');
    hold on;
    plot(S, 'c');
    hold on;
    for j = 1:length(p)
        if length(p{j}) == 3
            plot3(p{j}(1),p{j}(2),p{j}(3), '*', 'color', 'black');
            xlabel('x1');
            ylabel('x2');
            zlabel('x3');
        else
            plot(p{j}(1),p{j}(2), '*', 'color', 'black');
            xlabel('x1');
            ylabel('x2');
        end
    end
    % hold off;
    title('All');
end

fprintf('\nAbsdom bounds');
[l, u] = A.getRanges
fprintf('Zono bounds');
[l, u] = Z.getRanges;
fprintf('RStar Exact bounds');
[l, u] = R2.getExactRanges
fprintf('RStar approx bounds');
[l, u] = R2.getRanges
fprintf('Star absdom approx bounds');
[l, u] = Sa.getRanges
fprintf('Star approx bounds');
[l, u] = S.getRanges

function r = point_affineMap(p, W, b)
    r = W*p + b;
end

function r = point_reach(p)
    r = tansig(p);
end

function r = point_layerReach(p, W, b)
    r = tansig(W*p + b);
end