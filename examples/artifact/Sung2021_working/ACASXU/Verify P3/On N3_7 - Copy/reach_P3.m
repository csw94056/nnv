% ACASXU neural network verification with AbsDom, RStar, and Star sets
% excat-star, approx-star, absdom, rstar-two, rstar-three
% Sung Woo Choi: 12/28/2020

%% LOAD FFNN OBJECT
load ACASXU_run2a_3_7_batch_2000.mat;
Layers = [];
n = length(b);
for i=1:n - 1
    bi = cell2mat(b(i));
    Wi = cell2mat(W(i));
    Li = LayerS(Wi, bi, 'poslin');
    Layers = [Layers Li];
end
bn = cell2mat(b(n));
Wn = cell2mat(W(n));
Ln = LayerS(Wn, bn, 'purelin');
Layers = [Layers Ln];

% construct FFNN boject
F = FFNNS(Layers);

%% GENERATE INPUT SET
% Input Constraints
% 1500 <= i1(\rho) <= 1800,
% -0.06 <= i2 (\theta) <= 0.06,
% 3.1 <= i3 (\shi) <= 3.14
% 1000 <= i4 (\v_own) <= 1200, 
% 700 <= i5 (\v_in) <= 800

lb = [1500; -0.06; 3.1; 1000; 700];
ub = [1800; 0.06; 3.14; 1200; 800];

% normalize input
for i=1:5
    lb(i) = (lb(i) - means_for_scaling(i))/range_for_scaling(i);
    ub(i) = (ub(i) - means_for_scaling(i))/range_for_scaling(i);   
end

I_star = Star(lb, ub);
I_rstar = RStar(lb, ub, inf);
I_absdom = AbsDom(lb, ub, inf);

%% PERFORM REACHABILITY ANALYSIS
numCores = 8;

% [R0, t0] = F.reach(I_star.toPolyhedron, 'exact-polyhedron', numCores); % exact reach set using polyhedron
% F.print('F_exact_polyhedron.info'); % print all information to a file

[R1, t1] = F.reach(I_star, 'exact-star', numCores); % exact reach set using polyhdedron
F.print('F_exact_star.info'); % print all information to a file

[R2, t2] = F.reach(I_star, 'approx-star'); % approximate reach set using star
F.print('F_approx_star.info'); % print all information to a file

[R3, t3] = F.reach(I_star.getZono, 'approx-zono'); % approximate reach set using zonotope
F.print('F_approx_zono.info'); % print all information to a file

[R4, t4] = F.reach(I_star, 'abs-dom'); % approximate reach set using abstract domain
F.print('F_abs_dom.info'); % print all information to a file

[R5, t5] = F.reach(I_absdom, 'absdom'); % approximate reach set using absdom
F.print('F_absdom.info'); % print all information to a file

[R6, t6] = F.reach(I_rstar, 'rstar-absdom-two'); % approximate reach set using rstar with two predicate constraints
F.print('F_rstar_absdom_two.info'); % print all information to a file

[R7, t7] = F.reach(I_rstar, 'rstar-absdom-three'); % approximate reach set using rstar with three predicate constraints
F.print('F_rstar_absdom_three.info'); % print all information to a file

save outputSet.mat R1 R2 R3 R4 R5 R6 R7;
save outputTime.mat t1 t2 t3 t4 t5 t6 t7; 

