%% LOAD FFNN OBJECT
load NeuralNetwork7_3.mat;
Layers = [];
n = length(b);
for i=1:n
    bi = cell2mat(b(i));
    Wi = cell2mat(W(i));
    Li = LayerS(Wi, bi, 'poslin');
    Layers = [Layers Li];
end
% construct FFNN boject
F = FFNNS(Layers);

%% GENERATE INPUT SET
V = [zeros(3,1) eye(3)];
C = [eye(3); -eye(3)];
d = ones(6,1);
pred_lb = [-1; -1; -1];
pred_ub = [1; 1; 1];
I_star = Star(V, C, d, pred_lb, pred_ub); % input set as a Star set
I_absdom = AbsDom(I_star);
I_rstar = RStar(I_star);

%% PERFORM REACHABILITY ANALYSIS
% reachability parameters:
reachPRM_approx_star.inputSet = I_star;
reachPRM_approx_star.reachMethod = 'approx-star';
reachPRM_approx_star.numCores = 4;
reachPRM_approx_star.dis_opt = 'display';
reachPRM_approx_star.lp_solver = 'glpk';

reachPRM_absdom.inputSet = I_absdom;
reachPRM_absdom.reachMethod = 'absdom';
reachPRM_absdom.numCores = 4;
reachPRM_absdom.disp_opt = 'display';
reachPRM_absdom.lp_solver = 'glpk';

reachPRM_rstar_two.inputSet = I_rstar;
reachPRM_rstar_two.reachMethod = 'rstar-absdom-two';
reachPRM_rstar_two.numCores = 4;
reachPRM_rstar_two.disp_opt = 'display';
reachPRM_rstar_two.lp_solver = 'glpk';

reachPRM_rstar_three.inputSet = I_rstar;
reachPRM_rstar_three.reachMethod = 'rstar-absdom-three';
reachPRM_rstar_three.numCores = 4;
reachPRM_rstar_three.disp_opt = 'display';
reachPRM_rstar_three.lp_solver = 'glpk';

[R_approx_star, t_approx_star] = F.reach(reachPRM_approx_star); % compute reach set
[R_absdom, t_absdom] = F.reach(reachPRM_absdom);
[R_rstar_two, t_rstar_two] = F.reach(reachPRM_rstar_two);
[R_rstar_three, t_rstar_three] = F.reach(reachPRM_rstar_three);

% save F.mat F; % save the verified network
% F.print('F.info'); % print all information to a file


% generate some input to test the output
e = 0.25;
x = [];
y = [];
for x1=-1:e:1
    for x2=-1:e:1
        for x3=-1:e:1
            xi = [x1; x2; x3];
            yi = F.evaluate(xi);
            x = [x, xi];
            y = [y, yi];
        end
    end
end

fig = figure;
% Star.plots(R);
R_absdom.plot('y');
hold on;
R_rstar_two.plot('m');
hold on;
R_rstar_three.plot('g');
hold on;
R_approx_star.plot('r');
hold on;
plot(y(1, :), y(2, :), 'o');

