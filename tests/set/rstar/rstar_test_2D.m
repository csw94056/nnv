close all;
clear;
clc;

% RStar (relaxed star) test
% Sung Woo Choi: 03/25/2021

%-1 <= a[1] <= 1, -1 <= a[2] <= 1
V = [0 1 0; 0 0 1];
C = [eye(2); -eye(2)];
d = [1; 1; 1; 1];
lower_a = {[0 -1 0; 0 0 -1]};
upper_a = {[0  1 0; 0 0  1]};
pred_lb = -ones(2, 1);
pred_ub = ones(2, 1);
lb = {[-1; -1;]};
ub = {[1; 1]};
iter = inf;

R1 = RStar(V, C, d, pred_lb, pred_ub, lower_a, upper_a, lb, ub, iter);

W = [0.3 0.2; 0.2 0.3];
b = [0; 0];

R2 = R1.affineMap(W, b);


R3 = TanSig.reach_rstar_absdom_with_two_pred_const(R2);

figure;
R1.plot('r');
figure;
R2.plot('r');
figure;
R3.plot('r');

% lb = [-9; -3];
% ub = [1; 1];
% iter = inf;
% R1 = RStar(lb, ub, iter);
% % 

% 
% S1 = Star(lb, ub);
% 
% 
% 
% 
% %R2 = R1.affineMap(W, b);
% 
% 
% %hold on;
% %R2.plot('y');