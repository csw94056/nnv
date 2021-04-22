close all;
clear;
clc;
% 
% W{1} = [1 1; 1 -1];
% % W{2} = [1 1; 1 -1];
% % W{3} = [1 1; 1 0];
% % 
% b{1} = [0; 0];
% b{2} = [0; 1.5];
% b{3} = [1; 0];

for i = 1:3
%     W{i} = rand(2,2);
%     b{i} = rand(2,1);

    W{i} = -1 + 2*rand(2,2);
    b{i} = -1 + 2*rand(2,1);
    
%     W{i} = -10 + 20*rand(2,2);
%     b{i} = -10 + 20*rand(2,1);
    
%     W{i} = randi([0 5], 2,2);
%     b{i} = randi([0 5], 2,1);
end

% hidden layers have 3 nodes
% W{1} = rand(3,2);
% b{1} = rand(3,1);
% W{2} = rand(3,3);
% b{2} = rand(3,1);
% W{3} = rand(2,3);
% b{3} = rand(2,1);

% good example
% W{1} = [-5 -5; -2 -2];
% b{1} = [2; -3];
% W{2} = [2 5; 3 4];
% b{2} = [-2; 4];
% W{3} = [4 -2; 4 3];
% b{3} = [-2; 4];
% W{4} = [1 -5; -3 -1];
% b{4} = [0; -5];
% W{5} = [4 1; 3 5];
% b{5} = [2; 0];

% good example 2
% W{1} = [0.728214624538208 0.630757626740935; 0.542567725535353 0.269076315415997];
% W{2} = [0.925544898560559 0.066704479259003; 0.951119572011534 0.946137972467446];
% W{3} = [0.153641709762110 0.168748938145153; 0.296073114887304 0.735194614493405];
% b{1} = [0.591812006788566; 0.866305096716458];
% b{2} = [0.496633305608366; 0.068027386603233];
% b{3} = [0.879697065220770; 0.851423388162803];

% %good example 3
% W{1} = [0.465698898461883 0.343303486341360; 0.211809493190058 0.763338677110964];
% W{2} = [0.877803795161917 0.187516474910597; 0.151308867486537 0.949022459153999];
% W{3} = [0.424614495494384 0.852506608051607; 0.704501247596211 0.706492692448605];
% b{1} = [0.128698784708328; 0.015502586140148];
% b{2} = [0.948045881445746; 0.558770573875961];
% b{3} = [0.178788094416603; 0.305273656760658];

%example why AbsS is more conservative than exact-approx Star
% W{1} = [5 -1; 3 2];
% W{2} = [3 -2; 4 0];
% % W{3} = [-1 1; 1 0];
% % W{4} = [-3 -5; 4 -5];
% % W{5} = [1 5; -3 5];
% b{1} = [3; -2];
% b{2} = [2; 2];
% % b{3} = [0; 3];
% % b{4} = [0; -5];
% % b{5} = [1; 5];
%example2 why AbsS is more conservative than exact-approx Star
% W{1} = [-6.9348 9.5514; -2.4429 9.8134];
% W{2} = [2.5300 -2.6636; -6.8864 1.3701];
% W{3} = [-4.9032 7.8001; 8.3014 6.9620];
% W{4} = [-9.5327 -7.1667; 0.7005 8.5443];
% W{5} = [8.7960 -7.5174; 2.3997 -4.8236];
% b{1} = [-5.3579; -5.6636];
% b{2} = [-6.1321; 6.1095];
% b{3} = [-3.8838; -4.6882];
% b{4} = [-3.4347; -8.7425];
% b{5} = [1.5628; 9.7556];

%example3 why AbsS is more conservative than exact-approx Star
% W{1} = [-6 9; -2 9];
% W{2} = [2 -2; -6 1];
% W{3} = [-4 7; 8 6];
% W{4} = [-9 -7; 0 8];
% W{5} = [8 -7; 2 -4];
% b{1} = [-5; -5];
% b{2} = [-6; 6];
% b{3} = [-3; -4];
% b{4} = [-3; -8];
% b{5} = [1; 9];


% % W{1} = [1 -1; 1 -1];
% % W{2} = [1 1; 1 -1];
% W{3} = [1 1; 1 1];
% W{3} = [-1 1; 1 1];
%Ex 2 RlxS is more conservative than AbsS is conservative than Se
% W{1} = [1 -1; 1 -1];
% W{2} = [-1 1; -1 1];
% example AbsS more conservative
% W{1} = [1 -1; 1 -1];
% W{2} = [-1 1; -1 1];
% W{3} = [1 1; 1 -1];
% original example
% W{1} = [1 1; 1 -1];
% W{2} = [1 1; 1 -1];
% W{3} = [1 1; 0 1];
% b{1} = [0; 0]; 
% b{2} = [0; 0];
% b{3} = [0; 0];
% b{3} = [4; -1]
steps = length(W);


% P = Polyhedron('lb', [-1, 0], 'ub', [1, 0]);
P = Polyhedron('lb', [-1; -1], 'ub', [1; 1]);
% P = Polyhedron('lb', [0; 0], 'ub', [1; 1]);
% P = Polyhedron('lb', [0; 0], 'ub', [1; 1]);
A = AbsDom(P, inf);
R2 = RStar(P,inf);
R3 = RStar(P,inf);
Re = RStar(P,inf);
S = Star(P);
S.predicate_lb = [-1;-1];
S.predicate_ub = [1;1];
Sa = S;
figure;
nexttile;
plot(A, 'r');
hold on;
plot(R2, 'y');
hold on;
plot(Sa, 'b');
hold on
plot(S, 'c');          % Star approx
% hold on
% plot(Se, 'c');
% hold on;
% plot(R3, 'g');
% hold on;
% plot(Re, 'b');

title('input');
%legend('AbsDom','Star approx', 'RStar 2');
legend('AbsDom','RStar 2', 'Star AbsDom', 'Star approx');
[lb, ub] = R2.getRanges;
B = Box(lb, ub);
Z = B.toZono;
for i = 1:steps
    A = A.affineMap(W{i}, b{i});
    Sa = Sa.affineMap(W{i}, b{i});
    R2 = R2.affineMap(W{i}, b{i});
%     R3 = R3.affineMap(W{i}, b{i});
    %Re = Re.affineMap(W{i}, b{i});
    S = S.affineMap(W{i}, b{i});
    Z = Z.affineMap(W{i}, b{i});
    nexttile;
    plot(A, 'r');
    hold on;
    plot(R2, 'y');
    hold on;
    plot(Sa, 'b');
    hold on;
    plot(S, 'c');
%     hold on;
%     plot(Se, 'c');
    
%     plot(R3, 'g');
%     hold on;
%     plot(Re, 'b');
%     hold on;
%     plot(Z, 'b');
%     
    title('affine map');
    
    A = LogSig.reach_absdom_approx(A);
    Sa = LogSig.reach_abstract_domain_approx(Sa);
    R2 = LogSig.reach_rstar_absdom_with_two_pred_const(R2);
%     R3 = PosLin.reach_rstar_absdom_with_three_pred_const(R3, 0);
    %Re
    S = LogSig.reach_star_approx_no_split(S);
    [lb, ub] = R2.getRanges;
    B = Box(lb, ub);
    Z = B.toZono;

    nexttile;
    plot(A, 'r');
    hold on;
%     plot(Z, 'b');
%     hold on;
    plot(R2, 'y');
    hold on;
    plot(Sa, 'b');
    hold on;
    
%     plot(R3, 'g');
%     hold on;
    % plot(Re, 'b');
    hold on;
    plot(S, 'c');
    title('Activation Function');
end

figure;
nexttile;
plot(A, 'r');
hold on;
title('AbsDom');
nexttile;
plot(R2, 'y');
title('RStar 2');
nexttile;
plot(Sa, 'b');
title('Star AbsDom');
nexttile;
plot(S, 'c');
title('Star approx');

nexttile;
title('All');
plot(A, 'r');
hold on;
plot(R2, 'y');
hold on;
plot(Sa, 'b');
hold on;
plot(S, 'c');
% nexttile;
% plot(R3, 'g');
% title('RStar 3');

%{
T = R3;
T.lower_a{1} = [0 1 0; 0 0 1];

U = ub_backSub_exact(T, T.lower_a, T.upper_a)


u1 = [0 1 0; 0 0 1];
u2 = [-5 -6 9; -5 -2 9];
u3 = [20/3 1/3 0; 48/11 0 3/11];
%}




% lower bound back-substitution
function lb = lb_backSub_exact(obj, lower_a, upper_a)
    maxIter = obj.iter;
    len = length(upper_a);
    [nL, mL] = size(upper_a{len});
    alpha = upper_a{len}(:,2:end);
    lower_v = zeros(nL, 1);
    upper_v = upper_a{len}(:,1);

    % b[s+1] = v' + sum( max(0,w[j]')*lower_a[j] + min(w[j]',0)*upper_a[j}] ) for j is element of k and for k < i
    % iteration until lb' = b[s'] = v''
    len = len - 1;
    iter = 0;
    while (len > 1 && iter < maxIter)
        [nL, mL] = size(upper_a{len});
        dim = nL;

        max_a = max(0, alpha);
        min_a = min(alpha, 0);

        lower_v = max_a * lower_a{len}(:,1) + lower_v;
        upper_v = min_a * upper_a{len}(:,1) + upper_v;

        alpha = max_a * lower_a{len}(:,2:end) + ...
                min_a * upper_a{len}(:,2:end);

        len = len - 1;
        iter = iter + 1;
    end

    max_a = max(0, alpha);
    min_a = min(alpha, 0);

    [lb1,ub1] = getRanges_L(obj,len);
    lb = max_a * lb1 + lower_v + ...
         min_a * ub1 + upper_v;
end

% upper bound back-substituion
function ub = ub_backSub_exact(obj, lower_a, upper_a)
    maxIter = obj.iter;
    len = length(upper_a);
    [nL, mL] = size(upper_a{len});
    alpha = upper_a{len}(:,2:end);
    lower_v = zeros(nL, 1);
    upper_v = upper_a{len}(:,1);

    % c[t+1] = v' + sum( max(0,w[j]')*upper_a[j] + min(w[j]',0)*lower_a[j}] )  for j is element of k and for k < i
    % iteration until ub' = c[t'] = v''
    len = len - 1;
    iter = 0;
    while (len > 1 && iter < maxIter)
        dim = size(lower_a{len}, 1);

        max_a = max(0, alpha);
        min_a = min(alpha, 0);

        lower_v = min_a * lower_a{len}(:,1) + lower_v;
        upper_v = max_a * upper_a{len}(:,1) + upper_v;

        alpha = min_a * lower_a{len}(:,2:end) + ...
                max_a * upper_a{len}(:,2:end);

        len = len - 1;
        iter = iter + 1;
    end

    max_a = max(0, alpha);
    min_a = min(alpha, 0);

    [lb1,ub1] = getRanges_L(obj,len);
    ub = min_a * lb1 + lower_v + ...
         max_a * ub1 + upper_v;
end