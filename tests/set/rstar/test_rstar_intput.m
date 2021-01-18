% RStar (relaxed star) input test
% Sung Woo Choi: 12/18/2020

P = ExamplePoly.randHrep('d',2);
R1 = RStar(P, inf);
toP = R1.toPolyhedron;

figure;
nexttile;
P.plot;
title('Polyhedron');
nexttile;
R1.plot('y');
title('RStar');
nexttile;
toP.plot('color', 'g');
title('toPolyhedron');

c1 = [0; 0];
V1 = [1 -1; 1 1];
Z = Zono(c1, V1);
R2 = RStar(Z, inf);
toZ = R2.toZono;

figure;
nexttile;
Z.plot;
title('Zono');
nexttile;
R2.plot('y');
title('RStar');
nexttile;
toZ.plot('g');
title('toZono');

S = Star.rand(2);
R3 = RStar(S, inf);
toS = R3.toStar;
figure;
nexttile;
S.plot;
title('Star');
nexttile;
R3.plot('y');
title('RStar');
nexttile;
toS.plot('g');
title('toStar');