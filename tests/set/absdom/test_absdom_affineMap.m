P = ExamplePoly.randHrep('d',2);
A = AbsDom(P, inf);

W = [1 -1; 1 -1];
b = [0.5; 0.5];
A1 = A.affineMap(W, b);

figure;
A1.plot('y');
hold on;
A.plot('r');

