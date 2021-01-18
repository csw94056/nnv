load ACASXU_run2a_4_1_batch_2000.mat;
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
F = FFNNS(Layers);

% Input Constraints
% 1500 <= i1(\rho) <= 1800,
% -0.06 <= i2 (\theta) <= 0.06,
% \shi = 0
% 1000 <= i4 (\v_own) <= 1200, 
% 700 <= i5 (\v_in) <= 800

lb = [1500; -0.06; 0; 1000; 700];
ub = [1800; 0.06; 0; 1200; 800];

% normalize input
for i=1:5
    lb(i) = (lb(i) - means_for_scaling(i))/range_for_scaling(i);
    ub(i) = (ub(i) - means_for_scaling(i))/range_for_scaling(i);   
end

I = Star(lb, ub);
I_star = Star(lb, ub);
I_rstar = RStar(lb, ub, inf);
I_absdom = AbsDom(lb, ub, inf);

numCores = 8;

[R1, ~] = F.reach(I, 'exact-star', numCores); % exact reach set using polyhdedron
F.print('F_exact_star.info'); % print all information to a file

[R2, ~] = F.reach(I, 'approx-star'); % approximate reach set using star
F.print('F_approx_star.info'); % print all information to a file

[R3, ~] = F.reach(I.getZono, 'approx-zono'); % approximate reach set using zonotope
F.print('F_approx_zono.info'); % print all information to a file

[R4, ~] = F.reach(I, 'abs-dom'); % approximate reach set using abstract domain
F.print('F_abs_dom.info'); % print all information to a file

[R5, ~] = F.reach(I_absdom, 'absdom'); % approximate reach set using absdom (ERAN)
F.print('F_absdom.info'); % print all information to a file

[R6, ~] = F.reach(I_rstar, 'rstar-absdom-two'); % approximate reach set using rstar with abstract domain and two predicate constraints
F.print('F_rstar_absdom_two.info'); % print all information to a file

[R7, ~] = F.reach(I_rstar, 'rstar-absdom-three'); % approximate reach set using rstar with abstract domain and three predicate constraints
F.print('F_rstar_absdom_three.info'); % print all information to a file

[R8, ~] = F.reach(I_rstar, 'rstar-caseb-three'); % approximate reach set using rstar with case b abd three predicate constraints
F.print('F_rstar_absdom_three.info'); % print all information to a file

[R9, ~] = F.reach(I_rstar, 'rstar-casec-three'); % approximate reach set using rstar with case c three predicate constraints
F.print('F_rstar_absdom_three.info'); % print all information to a file

R10 = Star(R9.V, [R8.C; R9.C], [R8.d; R9.d]); 

save outputSet.mat R1 R2 R3 R4 R5 R6 R7 R8 R9;

normalized_mat = range_for_scaling(6) * eye(5);
normalized_vec = means_for_scaling(6) * ones(5,1);

% normalize output set

fprintf('\nNormalize output set');

n = length(R1);
R1_scaled = [];
for i=1:n
    R1_scaled = [R1_scaled  R1(i).affineMap(normalized_mat, normalized_vec)]; % exact normalized reach set
end
R2_scaled = R2.affineMap(normalized_mat, normalized_vec); % over-approximate normalized reach set using star
R3_scaled = R3.affineMap(normalized_mat, normalized_vec); % over-approximate normalized reach set using zonotope
R4_scaled = R4.affineMap(normalized_mat, normalized_vec); % over-approximate normalized reach set using abstract domain
R5_scaled = R5.affineMap(normalized_mat, normalized_vec); % over-approximate normalized reach set using absdom (ERAN)
R6_scaled = R6.affineMap(normalized_mat, normalized_vec); % over-approximate normalized reach set using rstar with abstract domain and two predicate constraints
R7_scaled = R7.affineMap(normalized_mat, normalized_vec); % over-approximate normalized reach set using abstract domain and three predicate constraints
R8_scaled = R8.affineMap(normalized_mat, normalized_vec); % over-approximate normalized reach set using case b abd three predicate constraints
R9_scaled = R9.affineMap(normalized_mat, normalized_vec); % over-approximate normalized reach set using case c abd three predicate constraints
R10_scaled = R10.affineMap(normalized_mat, normalized_vec); % R8 & R9

% output = [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
maps1 = [1 0 0 0 0; 0 1 0 0 0]; % plot a projection on COC, Weak Left
maps2 = [1 0 0 0 0; 0 0 1 0 0]; % plot a projection on COC, Weak Right
maps3 = [1 0 0 0 0; 0 0 0 1 0]; % plot a projection on COC, Strong Left
maps4 = [1 0 0 0 0; 0 0 0 0 1]; % plot a projection on COC, Strong Right

R11 = [];
R12 = [];
R13 = [];
R14 = [];
for i=1:n
    R11 = [R11 R1_scaled(i).affineMap(maps1, [])];
    R12 = [R12 R1_scaled(i).affineMap(maps2, [])];
    R13 = [R13 R1_scaled(i).affineMap(maps3, [])];
    R14 = [R14 R1_scaled(i).affineMap(maps4, [])];
end

R21 = R2_scaled.affineMap(maps1, []);
R22 = R2_scaled.affineMap(maps2, []);
R23 = R2_scaled.affineMap(maps3, []);
R24 = R2_scaled.affineMap(maps4, []);

R31 = R3_scaled.affineMap(maps1, []);
R32 = R3_scaled.affineMap(maps2, []);
R33 = R3_scaled.affineMap(maps3, []);
R34 = R3_scaled.affineMap(maps4, []);

R41 = R4_scaled.affineMap(maps1, []);
R42 = R4_scaled.affineMap(maps2, []);
R43 = R4_scaled.affineMap(maps3, []);
R44 = R4_scaled.affineMap(maps4, []);

R51 = R5_scaled.affineMap(maps1, []);
R52 = R5_scaled.affineMap(maps2, []);
R53 = R5_scaled.affineMap(maps3, []);
R54 = R5_scaled.affineMap(maps4, []);

R61 = R6_scaled.affineMap(maps1, []);
R62 = R6_scaled.affineMap(maps2, []);
R63 = R6_scaled.affineMap(maps3, []);
R64 = R6_scaled.affineMap(maps4, []);

R71 = R7_scaled.affineMap(maps1, []);
R72 = R7_scaled.affineMap(maps2, []);
R73 = R7_scaled.affineMap(maps3, []);
R74 = R7_scaled.affineMap(maps4, []);

R81 = R8_scaled.affineMap(maps1, []);
R82 = R8_scaled.affineMap(maps2, []);
R83 = R8_scaled.affineMap(maps3, []);
R84 = R8_scaled.affineMap(maps4, []);

R91 = R9_scaled.affineMap(maps1, []);
R92 = R9_scaled.affineMap(maps2, []);
R93 = R9_scaled.affineMap(maps3, []);
R94 = R9_scaled.affineMap(maps4, []);

R101 = R10_scaled.affineMap(maps1, []);
R102 = R10_scaled.affineMap(maps2, []);
R103 = R10_scaled.affineMap(maps3, []);
R104 = R10_scaled.affineMap(maps4, []);

% plot reachable set

fprintf('\nPlotting reachable set...');

fig = figure;
subplot(1, 4, 1);
% R31.plot; %Zonotope
% hold on;
% R101.plot; %R9 and R8 case b and case c
% hold on;
% R91.plot; %RStar case c
% hold on;
% R81.plot; %RStar case b
% hold on;
R51.plot; %RStar absdom
hold on;
R41.plot; %Star abs-dom
hold on;
% R61.plot; %RStar abs-dom two
% hold on;
R71.plot; %RStar abs-dom three
hold on;
R21.plot;
hold on;
Star.plots(R11);

xlabel('COC', 'Fontsize', 16);
ylabel('Weak-Left', 'Fontsize', 16);
set(gca, 'Fontsize', 16);

subplot(1, 4, 2);
% R32.plot; %Zonotope
% hold on;
% R102.plot; %R9 and R8 case b and case c
% hold on;
% R92.plot; %RStar case c
% hold on;
% R82.plot; %RStar case b
% hold on;
R52.plot; %RStar absdom
hold on;
R42.plot; %Star abs-dom
hold on;
% R62.plot; %RStar abs-dom two
% hold on;
R72.plot; %RStar abs-dom three
hold on;
R22.plot;
hold on;
Star.plots(R12);
xlabel('COC', 'Fontsize', 16);
ylabel('Weak-Right', 'Fontsize', 16);
set(gca, 'Fontsize', 16);

subplot(1, 4, 3);
% R33.plot; %Zonotope
% hold on;
% R103.plot; %R9 and R8 case b and case c
% hold on;
% R93.plot; %RStar case c
% hold on;
% R83.plot; %RStar case b
% hold on;
R53.plot; %RStar absdom
hold on;
R43.plot; %Star abs-dom
hold on;
% R63.plot; %RStar abs-dom two
% hold on;
R73.plot; %RStar abs-dom three
hold on;
R23.plot;
hold on;
Star.plots(R13);
xlabel('COC', 'Fontsize', 16);
ylabel('Weak-Right', 'Fontsize', 16);
set(gca, 'Fontsize', 16);

subplot(1, 4, 4);
% R34.plot; %Zonotope
% hold on;
% R104.plot; %R9 and R8 case b and case c
% hold on;
% R94.plot; %RStar case c
% hold on;
% R84.plot; %RStar case b
% hold on;
R54.plot; %RStar absdom
hold on;
R44.plot; %Star abs-dom
hold on;
% R64.plot; %RStar abs-dom two
% hold on;
R74.plot; %RStar abs-dom three
hold on;
R24.plot;
hold on;
Star.plots(R14);
xlabel('COC', 'Fontsize', 16);
ylabel('Strong-Right', 'Fontsize', 16);
set(gca, 'Fontsize', 16);


set(gca, 'Fontsize', 16);
saveas(gcf, 'reachSet_P4_on_N4_1.pdf');
