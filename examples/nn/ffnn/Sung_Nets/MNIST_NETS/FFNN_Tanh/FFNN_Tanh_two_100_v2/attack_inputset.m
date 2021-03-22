load images.mat;
load MNIST_tanh_100_100_DenseNet.mat;
N = size(IM_labels, 1); % number of test images used to testing robustness

figure                                          % initialize figure
colormap(gray)                                  % set to grayscale
for i = 1:50                                    % preview first 36 samples
    subplot(5,10,i)                              % plot them in 6 x 6 grid
    digit = reshape(IM_data(:,i), [28,28]);     % row = 28 x 28 image
    imagesc(digit)                              % show the image
    title(IM_labels(i))                   % show the label
end


% image perturbation by infinity norm attack (offsetting images)
% eps = 0.005;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_005(n) = Star(lb, ub);
%    A_eps_005(n) = AbsDom(lb, ub, inf);
%    RS_eps_005(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.010;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_010(n) = Star(lb, ub);
%    A_eps_010(n) = AbsDom(lb, ub, inf);
%    RS_eps_010(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.015;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_015(n) = Star(lb, ub);
%    A_eps_015(n) = AbsDom(lb, ub, inf);
%    RS_eps_015(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.020;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_020(n) = Star(lb, ub);
%    A_eps_020(n) = AbsDom(lb, ub, inf);
%    RS_eps_020(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.025;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_025(n) = Star(lb, ub);
%    A_eps_025(n) = AbsDom(lb, ub, inf);
%    RS_eps_025(n) = RStar(lb, ub, inf);
% end
% 
% eps = 0.030;
% for n = 1:N
%    IM = IM_data(:, n);
%    lb = IM - eps;
%    ub = IM + eps;
%    S_eps_030(n) = Star(lb, ub);
%    A_eps_030(n) = AbsDom(lb, ub, inf);
%    RS_eps_030(n) = RStar(lb, ub, inf);
% end

eps = 0.1
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_1(n) = Star(lb, ub);
   A_eps_1(n) = AbsDom(lb, ub, inf);
   RS_eps_1(n) = RStar(lb, ub, inf);
end

eps = 0.2
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_2(n) = Star(lb, ub);
   A_eps_2(n) = AbsDom(lb, ub, inf);
   RS_eps_2(n) = RStar(lb, ub, inf);
end

eps = 0.3
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_3(n) = Star(lb, ub);
   A_eps_3(n) = AbsDom(lb, ub, inf);
   RS_eps_3(n) = RStar(lb, ub, inf);
end

eps = 0.4
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_4(n) = Star(lb, ub);
   A_eps_4(n) = AbsDom(lb, ub, inf);
   RS_eps_4(n) = RStar(lb, ub, inf);
end

eps = 0.5
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_5(n) = Star(lb, ub);
   A_eps_5(n) = AbsDom(lb, ub, inf);
   RS_eps_5(n) = RStar(lb, ub, inf);
end

eps = 0.6
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_6(n) = Star(lb, ub);
   A_eps_6(n) = AbsDom(lb, ub, inf);
   RS_eps_6(n) = RStar(lb, ub, inf);
end

eps = 0.7
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_7(n) = Star(lb, ub);
   A_eps_7(n) = AbsDom(lb, ub, inf);
   RS_eps_7(n) = RStar(lb, ub, inf);
end

eps = 0.8
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_8(n) = Star(lb, ub);
   A_eps_8(n) = AbsDom(lb, ub, inf);
   RS_eps_8(n) = RStar(lb, ub, inf);
end

eps = 0.9
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_9(n) = Star(lb, ub);
   A_eps_9(n) = AbsDom(lb, ub, inf);
   RS_eps_9(n) = RStar(lb, ub, inf);
end

eps = 1.0
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_10(n) = Star(lb, ub);
   A_eps_10(n) = AbsDom(lb, ub, inf);
   RS_eps_10(n) = RStar(lb, ub, inf);
end

% save not_normalized/inputStar005.mat S_eps_005
% save not_normalized/inputStar010.mat S_eps_010 
% save not_normalized/inputStar015.mat S_eps_015  
% save not_normalized/inputStar020.mat S_eps_020 
% save not_normalized/inputStar025.mat S_eps_025 
save not_normalized/inputStar1.mat S_eps_1
save not_normalized/inputStar2.mat S_eps_2
save not_normalized/inputStar3.mat S_eps_3
save not_normalized/inputStar4.mat S_eps_4
save not_normalized/inputStar5.mat S_eps_5
save not_normalized/inputStar6.mat S_eps_6
save not_normalized/inputStar7.mat S_eps_7
save not_normalized/inputStar8.mat S_eps_8
save not_normalized/inputStar9.mat S_eps_9
save not_normalized/inputStar10.mat S_eps_10

% save not_normalized/inputAbsDom005.mat A_eps_005 
% save not_normalized/inputAbsDom010.mat A_eps_010 
% save not_normalized/inputAbsDom015.mat A_eps_015 
% save not_normalized/inputAbsDom020.mat A_eps_020  
% save not_normalized/inputAbsDom025.mat A_eps_025 
% save not_normalized/inputAbsDom030.mat A_eps_030
save not_normalized/inputAbsDom1.mat A_eps_1
save not_normalized/inputAbsDom2.mat A_eps_2
save not_normalized/inputAbsDom3.mat A_eps_3
save not_normalized/inputAbsDom4.mat A_eps_4
save not_normalized/inputAbsDom5.mat A_eps_5
save not_normalized/inputAbsDom6.mat A_eps_6
save not_normalized/inputAbsDom7.mat A_eps_7
save not_normalized/inputAbsDom8.mat A_eps_8
save not_normalized/inputAbsDom9.mat A_eps_9
save not_normalized/inputAbsDom10.mat A_eps_10

% save not_normalized/inputRStar005.mat RS_eps_005 
% save not_normalized/inputRStar010.mat RS_eps_010 
% save not_normalized/inputRStar015.mat RS_eps_015 
% save not_normalized/inputRStar020.mat RS_eps_020 
% save not_normalized/inputRStar025.mat RS_eps_025 
% save not_normalized/inputRStar030.mat RS_eps_030
save not_normalized/inputRStar1.mat RS_eps_1
save not_normalized/inputRStar2.mat RS_eps_2
save not_normalized/inputRStar3.mat RS_eps_3
save not_normalized/inputRStar4.mat RS_eps_4
save not_normalized/inputRStar5.mat RS_eps_5
save not_normalized/inputRStar6.mat RS_eps_6
save not_normalized/inputRStar7.mat RS_eps_7
save not_normalized/inputRStar8.mat RS_eps_8
save not_normalized/inputRStar9.mat RS_eps_9
save not_normalized/inputRStar10.mat RS_eps_10