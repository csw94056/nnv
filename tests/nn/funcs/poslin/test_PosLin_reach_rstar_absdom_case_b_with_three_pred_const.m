
P = ExamplePoly.randHrep('d', 2);
P.outerApprox;
Is = Star(P);
Is.predicate_lb = P.Internal.lb;
Is.predicate_ub = P.Internal.ub;

Ir = RStar(P);
X = Is.sample(100);

figure;
Ir.plot;
hold on;
plot(X(1,:), X(2,:), 'ob'); % sampled inputs


R2 = PosLin.reach_rstar_absdom_with_two_pred_const(Ir); % over-approximate abstract domain reach set
R3 = PosLin.reach_rstar_absdom_case_b_with_three_pred_const(Ir); % over-approximate abstract domain reach set
S = PosLin.reach_star_approx(Is); % over-approximate reach set

Y = PosLin.evaluate(X);

figure;
nexttile;
R2.plot;
hold on;
R3.plot('y');
hold on;
Star.plots(S,'m');
hold on;
plot(Y(1,:), Y(2,:), '*'); % sampled outputs
nexttile;
R3.plot('y');
title('RStar absdom 3Const');
nexttile;
Star.plots(S,'m');
title('Approx Star');