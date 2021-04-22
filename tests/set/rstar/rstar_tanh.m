close all;
clear;
clc;

% W{1} = [-0.4350 0.1897; -0.0677 0.0066];
% W{2} = [0.2060 0.2058; 0.4152 -0.3669];
% W{3} = [0.0024 0.4604; -0.1010 -0.3843];
% W{4} = [-0.3411 -0.1803; 0.0252 0.2813];
% b{1} = [0.2169; -0.1988];
% b{2} = [0.1655; -0.1244];
% b{3} = [0.0612; -0.0328];
% b{4} = [-0.4371; 0.1223];

% W{1} = [0.0683 0.2682; 0.4670 0.4761];
% W{2} = [0.3553 0.0552; -0.4307 0.1719];
% W{3} = [-0.2959 0.3593; -0.3156 -0.4770];
% b{1} = [-0.2637; -0.1543];
% b{2} = [-0.0053; 0.1540];
% b{3} = [-0.2700; 0.2782];

% W{1} = [-0.0671 -0.3012; 0.3924 0.2009];
% W{2} = [0.1715 -0.3433; -0.0858 0.0986];
% W{3} = [0.3610 -0.0508; 0.0551 -0.4244];
% b{1} = [-0.1122; -0.1300];
% b{2} = [-0.0040; -0.0036];
% b{3} = [0.4846; 0.0233];

W{1} = [0.4655 -0.4823; -0.3404 0.3514];
W{2} = [0.2169 -0.1982; -0.3738 -0.4230];
W{3} = [-0.1741 -0.2249; 0.4896 -0.4516];
b{1} = [0.2233; 0.3517];
b{2} = [-0.3748; 0.3517];
b{3} = [0.2714; 0.3402];

% for i = 1:3
%     W{i} = -0.5 + rand(2,2);
%     b{i} = -0.5 + rand(2,1);
% end
steps = length(W);

P = Polyhedron('lb', [-1; -1], 'ub', [1; 1]);
A = AbsDom(P, inf);
R = RStar(P,inf);
S = Star(P);
S.predicate_lb = [-1;-1];
S.predicate_ub = [1;1];

figure;
nexttile;
plot(A, 'r');
hold on;
plot(S, 'b');
hold on;
plot(R, 'y');

title('input');
legend('AbsDom','Approx Star', 'RStar');

for i = 1:steps
    A = A.affineMap(W{i}, b{i});
    S = S.affineMap(W{i}, b{i});
    R = R.affineMap(W{i}, b{i});
    

    nexttile;
    plot(A, 'r');
    hold on;
    plot(R, 'y');
    hold on;
    plot(S, 'b');
    title('affine map');
    
    A = TanSig.reach_absdom_approx(A);
    S = TanSig.multiStepTanSig_NoSplit(S, 0, 'glpk');
    R = TanSig.reach_rstar_absdom_with_two_pred_const(R);


    nexttile;
    plot(A, 'r');
    hold on;
    plot(R, 'y');
    hold on;
    plot(S, 'b');
    hold on;
    legend('AbsDom','Approx Star', 'RStar');
    title('Activation Function');
end
figure;
nexttile;
plot(A, 'r');
hold on;
plot(R, 'y');
hold on;
plot(S, 'b');
hold on;
legend('AbsDom','Approx Star', 'RStar');
title('Reachable Sets');

figure;
nexttile;
plot(A, 'r');
hold on;
title('AbsDom');
nexttile;
plot(S, 'b');
title('Star');
nexttile;
plot(R, 'y');
title('RStar');
