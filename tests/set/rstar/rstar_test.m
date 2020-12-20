 % -1 <= a[1] <= 1, -1 <= a[2] <= 2, -2 <= a[3] <= 3
V = [0 1 0 0; 0 0 1 0; 0 0 0 1];
C = [eye(3); -eye(3)];
d = [1; 2; 3; 1; 1; 2];
lower_a = {[0 -1 0 0; 0 0 -1 0; 0 0 0 -2]};
upper_a = {[0 1 0 0; 0 0 2 0; 0 0 0 3]};
lb = {[-1; -1; -2]};
ub = {[1; 2; 3]};
iter = inf;

R1 = RStar(V, C, d, lower_a, upper_a, lb, ub, iter);

W = [2 1 1; 1 0 2];
b = [0.5; 0.5];

R2 = R1.affineMap(W, b);

figure;
R1.plot('r');
hold on;
R2.plot('y');