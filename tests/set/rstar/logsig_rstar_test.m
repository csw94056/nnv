close all;
clear;
clc;

for i = 1:5
%     W{i} = rand(2,2);
%     b{i} = rand(2,1);

    W{i} = -1 + 2*rand(2,2);
    b{i} = -1 + 2*rand(2,1);
    
%     W{i} = -10 + 20*rand(2,2);
%     b{i} = -10 + 20*rand(2,1);
    
%     W{i} = randi([0 5], 2,2);
%     b{i} = randi([0 5], 2,1);
end
save Wb2_logsig.mat W b;
% load Wb2_logsig.mat
steps = length(W);

% P = Polyhedron('lb', [-1, 0], 'ub', [1, 0]);
P = Polyhedron('lb', [-1; -1], 'ub', [1; 1]);
% P = Polyhedron('lb', [0; 0], 'ub', [1; 1]);
% P = Polyhedron('lb', [0; 0], 'ub', [1; 1]);
A = AbsDom(P, inf);
R2 = RStar(P,inf);
R3 = RStar(P,inf);
Re = RStar(P,inf);
S = Star(P);
S.predicate_lb = [-1;-1];
S.predicate_ub = [1;1];
Sa = S;
figure;
nexttile;
plot(A, 'r');
hold on;
plot(R2, 'y');
hold on;
plot(Sa, 'b');
hold on
plot(S, 'c');          % Star approx
% hold on
% plot(Se, 'c');
% hold on;
% plot(R3, 'g');
% hold on;
% plot(Re, 'b');

title('input');
%legend('AbsDom','Star approx', 'RStar 2');
legend('AbsDom','RStar 2', 'Star AbsDom', 'Star approx');
[lb, ub] = R2.getRanges;
B = Box(lb, ub);
Z = B.toZono;
for i = 1:steps
    A = A.affineMap(W{i}, b{i});
    Sa = Sa.affineMap(W{i}, b{i});
    R2 = R2.affineMap(W{i}, b{i});
%     R3 = R3.affineMap(W{i}, b{i});
    %Re = Re.affineMap(W{i}, b{i});
    S = S.affineMap(W{i}, b{i});
    Z = Z.affineMap(W{i}, b{i});
    nexttile;
    plot(A, 'r');
    hold on;
    plot(R2, 'y');
    hold on;
    plot(Sa, 'b');
%     hold on;
%     plot(S, 'c');
%     hold on;
%     plot(Se, 'c');
    
%     plot(R3, 'g');
%     hold on;
%     plot(Re, 'b');
%     hold on;
%     plot(Z, 'b');
%     
    title('affine map');
    
    A = LogSig.reach_absdom_approx(A);
    Sa = LogSig.reach_abstract_domain_approx(Sa);
    R2 = LogSig.reach_rstar_absdom_with_two_pred_const(R2);
%     R3 = PosLin.reach_rstar_absdom_with_three_pred_const(R3, 0);
    %Re
    S = LogSig.reach_star_approx_no_split(S);
    [lb, ub] = R2.getRanges;
    B = Box(lb, ub);
    Z = B.toZono;

    nexttile;
    plot(A, 'r');
    hold on;
%     plot(Z, 'b');
%     hold on;
    plot(R2, 'y');
    hold on;
    plot(Sa, 'b');
    hold on;
%     plot(R3, 'g');
%     hold on;
    % plot(Re, 'b');
%     hold on;
%     plot(S, 'c');
    title('Activation Function');
end

figure;
nexttile;
plot(A, 'r');
hold on;
title('AbsDom')
nexttile;
plot(R2, 'y');
title('RStar 2');;
nexttile;
plot(Sa, 'b');
title('Star AbsDom');
% nexttile;
% plot(S, 'c');
% title('Star approx');

nexttile;
title('All');
plot(A, 'r');
hold on;
plot(R2, 'y');
hold on;
% plot(Sa, 'b');
% hold on;
% plot(S, 'c');
% nexttile;
% plot(R3, 'g');
% title('RStar 3');
