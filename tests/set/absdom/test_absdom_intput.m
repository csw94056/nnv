% AbsDom input test
% Sung Woo Choi: 12/18/2020

P = ExamplePoly.randHrep('d',2);
A1 = AbsDom(P, inf);
toP = A1.toPolyhedron;

figure;
nexttile;
P.plot;
title('Polyhedron');
nexttile;
A1.plot('y');
title('AbsDom');
nexttile;
toP.plot('color', 'g');
title('toPolyhedron');

c1 = [0; 0];
V1 = [1 -1; 1 1];
Z = Zono(c1, V1);
A2 = AbsDom(Z, inf);
toZ = A2.toZono;

figure;
nexttile;
Z.plot;
title('Zono');
nexttile;
A2.plot('y');
title('AbsDom');
nexttile;
toZ.plot('g');
title('toZono');

S = Star.rand(2);
A3 = AbsDom(S, inf);
toS = A3.toStar;
figure;
nexttile;
S.plot;
title('Star');
nexttile;
A3.plot('y');
title('AbsDom');
nexttile;
toS.plot('g');
title('toStar');