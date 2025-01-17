
I = ExamplePoly.randVrep;   
V = [0 0; 1 0; 0 1];
I.outerApprox;
lb = I.Internal.lb;
ub = I.Internal.ub;
I = Star(V', I.A, I.b, lb, ub); % input star
X = I.sample(100);

figure;
I.plot;
hold on;
plot(X(1, :), X(2, :), 'ob'); % sampled inputs

S1 = SatLins.reach(I, 'exact-star'); % exach reach set using star
S2 = SatLins.reach(I, 'approx-star'); % over-approximate reach set using star
S3 = SatLins.reach(I, 'abs-dom'); % over-approximate reach set using abstract-domain


Y = SatLins.evaluate(X);

figure;
S3.plot;
hold on;
S2.plot;
hold on;
Star.plots(S1);
plot(Y(1, :), Y(2, :), '*'); % sampled outputs

