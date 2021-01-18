% RStar (relaxed star) affine transformer test
% Sung Woo Choi: 12/18/2020

P = ExamplePoly.randHrep('d',2);
R = RStar(P, inf);

W = [1 -1; 1 -1];
b = [0.5; 0.5];
R1 = R.affineMap(W, b);

figure;
R1.plot('y');
hold on;
R.plot('r');

