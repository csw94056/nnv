
I = ExamplePoly.randVrep;   
V = [0 0; 1 0; 0 1];
I.outerApprox;
lb = I.Internal.lb;
ub = I.Internal.ub;
I = Star(V', I.A, I.b, lb, ub); % input star

figure;
I.plot;
S = SatLin.stepReach(I, 1);
figure;
Star.plots(S);