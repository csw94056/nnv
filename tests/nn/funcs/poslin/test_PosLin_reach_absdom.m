P = ExamplePoly.randHrep('d', 2);
P.outerApprox;
Is = Star(P);
Is.predicate_lb = P.Internal.lb;
Is.predicate_ub = P.Internal.ub;

Ia = AbsDom(P);
X = Is.sample(100);

figure;
Ia.plot;
hold on;
Is.plot('m');
hold on;
plot(X(1,:), X(2,:), 'ob'); % sampled inputs

A = PosLin.reach_absdom(Ia); % over-approximate abstract domain reach set
S = PosLin.reach_star_approx(Is); % over-approximate reach set

Y = PosLin.evaluate(X);

figure;
A.plot;
hold on;
Star.plots(S,'m');
hold on;
plot(Y(1,:), Y(2,:), '*'); % sampled outputs