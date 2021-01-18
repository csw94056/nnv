% AbsDom test
% Sung Woo Choi: 12/18/2020
    
% -1 <= a[1] <= 1, -1 <= a[2] <= 2, -2 <= a[3] <= 3
lower_a = {[0 -1 0 0; 0 0 -1 0; 0 0 0 -2]};
upper_a = {[0 1 0 0; 0 0 2 0; 0 0 0 3]};
lb = {[-1; -1; -2]};
ub = {[1; 2; 3]};
iter = inf;

A1 = AbsDom(lower_a, upper_a, lb, ub, iter);

W = [2 1 1; 1 0 2];
b = [0.5; 0.5];

A2 = A1.affineMap(W, b);

figure;
A1.plot('r');
hold on;
A2.plot('y');