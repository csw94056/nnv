close all;
clear;
clc;
format long;

input_dim = 4;
output_dim = 3;
% n is number of neurons per hidden layers
n = 2;
% h+2 is number of hidden layers
h = 3;

W{1} = -1 + 2*rand(n,input_dim);
b{1} = -1 + 2*rand(n, 1);
for i = 2:4
    W{i} = -1 + 2*rand(n,n);
    b{i} = -1 + 2*rand(n,1);
end
W{i+1} = -1 + 2*rand(output_dim, n);
b{i+1} = -1 + 2*rand(output_dim, 1);

% save Wb2_tansig.mat W b;
% load Wb2_tansig.mat
steps = length(W);
% steps = 5;


P = Polyhedron('lb', -ones(input_dim, 1), 'ub', ones(input_dim, 1));
A = AbsDom(P, inf);
R2 = RStar(P,inf);
S = Star(P);
S.predicate_lb = -ones(input_dim, 1);
S.predicate_ub = ones(input_dim, 1);
Sa = S;
% figure;
% nexttile;
% plot(A, 'r');
% hold on;
% plot(Sa, 'b');
% hold on;
% plot(R2, 'y');
% hold on
% plot(S, 'c');          % Star approx
% title('input');

% %legend('AbsDom','Star approx', 'RStar 2');
% legend('AbsDom', 'Star AbsDom', 'RStar 2', 'Star approx');

for i = 1:steps
    A = A.affineMap(W{i}, b{i});
    Sa = Sa.affineMap(W{i}, b{i});
    R2 = R2.affineMap(W{i}, b{i});
    S = S.affineMap(W{i}, b{i});
%     nexttile;
%     plot(A, 'r');
%     hold on;
%     plot(Sa, 'b');
%     hold on;
%     plot(R2, 'y');
%     hold on;
%     plot(S, 'c');
%     title('affine map');
    
    A = TanSig.reach_absdom_approx(A);
    Sa = TanSig.reach_abstract_domain_approx(Sa);
    R2 = TanSig.reach_rstar_absdom_with_two_pred_const(R2);
    S = TanSig.multiStepTanSig_NoSplit(S);

    
%     nexttile;
%     plot(A, 'r');
%     hold on;
%     plot(Sa, 'b');
%     hold on;
%     plot(R2, 'y');
%     hold on;
%     plot(S, 'c');
%     title('Activation Function');
end

% figure;
% nexttile;
% plot(A, 'r');
% hold on;
% title('AbsDom');
% nexttile;
% plot(Sa, 'b');
% title('Star AbsDom');
% nexttile;
% plot(R2, 'y');
% title('RStar 2');
% nexttile;
% plot(S, 'c');
% title('Star approx');

% nexttile;
% title('All');
% plot(A, 'r');
% hold on;
% plot(Sa, 'b');
% hold on;
% plot(R2, 'y');
% hold on;
% plot(S, 'c');
% % nexttile;
% % plot(R3, 'g');
% % title('RStar 3');

fprintf('Absdom bounds;');
[l, u] = A.getRanges
fprintf('RStar Exact bounds');
[l, u] = R2.getExactRanges
fprintf('RStar approx bounds');
[l, u] = R2.getRanges
fprintf('Star absdom approx bounds');
[l, u] = Sa.getRanges
fprintf('Star approx bounds');
[l, u] = S.getRanges


% figure;
% nexttile;
% plot(A, 'r');
% hold on;
% title('AbsDom');
% nexttile;
% plot(Sa, 'b');
% title('Star AbsDom');
% nexttile;
% plot(R2, 'y');
% title('RStar 2');
% nexttile;
% plot(S, 'c');
% title('Star approx');

