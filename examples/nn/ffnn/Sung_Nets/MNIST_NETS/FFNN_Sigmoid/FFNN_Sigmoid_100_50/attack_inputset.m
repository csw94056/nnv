load sigmoid_100_50_images.mat;
load MNIST_sigmoid_100_50_DenseNet.mat;
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
eps = 0.2
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_02(n) = Star(lb, ub);
   A_eps_02(n) = AbsDom(lb, ub, inf);
   RS_eps_02(n) = RStar(lb, ub, inf);
end

eps = 0.4
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_04(n) = Star(lb, ub);
   A_eps_04(n) = AbsDom(lb, ub, inf);
   RS_eps_04(n) = RStar(lb, ub, inf);
end

eps = 0.6
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_06(n) = Star(lb, ub);
   A_eps_06(n) = AbsDom(lb, ub, inf);
   RS_eps_06(n) = RStar(lb, ub, inf);
end

eps = 0.8
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_08(n) = Star(lb, ub);
   A_eps_08(n) = AbsDom(lb, ub, inf);
   RS_eps_08(n) = RStar(lb, ub, inf);
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

eps = 1.2
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_12(n) = Star(lb, ub);
   A_eps_12(n) = AbsDom(lb, ub, inf);
   RS_eps_12(n) = RStar(lb, ub, inf);
end

eps = 1.4
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_14(n) = Star(lb, ub);
   A_eps_14(n) = AbsDom(lb, ub, inf);
   RS_eps_14(n) = RStar(lb, ub, inf);
end

eps = 1.6
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_16(n) = Star(lb, ub);
   A_eps_16(n) = AbsDom(lb, ub, inf);
   RS_eps_16(n) = RStar(lb, ub, inf);
end

eps = 1.8
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_18(n) = Star(lb, ub);
   A_eps_18(n) = AbsDom(lb, ub, inf);
   RS_eps_18(n) = RStar(lb, ub, inf);
end

eps = 2.0
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_20(n) = Star(lb, ub);
   A_eps_20(n) = AbsDom(lb, ub, inf);
   RS_eps_20(n) = RStar(lb, ub, inf);
end

eps = 2.2
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_22(n) = Star(lb, ub);
   A_eps_22(n) = AbsDom(lb, ub, inf);
   RS_eps_22(n) = RStar(lb, ub, inf);
end

eps = 2.4
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_24(n) = Star(lb, ub);
   A_eps_24(n) = AbsDom(lb, ub, inf);
   RS_eps_24(n) = RStar(lb, ub, inf);
end

eps = 2.6
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_26(n) = Star(lb, ub);
   A_eps_26(n) = AbsDom(lb, ub, inf);
   RS_eps_26(n) = RStar(lb, ub, inf);
end

save sigmoid_100_50_not_normalized/inputStar02.mat S_eps_02
save sigmoid_100_50_not_normalized/inputStar04.mat S_eps_04
save sigmoid_100_50_not_normalized/inputStar06.mat S_eps_06
save sigmoid_100_50_not_normalized/inputStar08.mat S_eps_08
save sigmoid_100_50_not_normalized/inputStar10.mat S_eps_10
save sigmoid_100_50_not_normalized/inputStar12.mat S_eps_12
save sigmoid_100_50_not_normalized/inputStar14.mat S_eps_14
save sigmoid_100_50_not_normalized/inputStar16.mat S_eps_16
save sigmoid_100_50_not_normalized/inputStar18.mat S_eps_18
save sigmoid_100_50_not_normalized/inputStar20.mat S_eps_20
save sigmoid_100_50_not_normalized/inputStar22.mat S_eps_22
save sigmoid_100_50_not_normalized/inputStar24.mat S_eps_24
save sigmoid_100_50_not_normalized/inputStar26.mat S_eps_26

save sigmoid_100_50_not_normalized/inputAbsDom02.mat A_eps_02
save sigmoid_100_50_not_normalized/inputAbsDom04.mat A_eps_04
save sigmoid_100_50_not_normalized/inputAbsDom06.mat A_eps_06
save sigmoid_100_50_not_normalized/inputAbsDom08.mat A_eps_08
save sigmoid_100_50_not_normalized/inputAbsDom10.mat A_eps_10
save sigmoid_100_50_not_normalized/inputAbsDom12.mat A_eps_12
save sigmoid_100_50_not_normalized/inputAbsDom14.mat A_eps_14
save sigmoid_100_50_not_normalized/inputAbsDom16.mat A_eps_16
save sigmoid_100_50_not_normalized/inputAbsDom18.mat A_eps_18
save sigmoid_100_50_not_normalized/inputAbsDom20.mat A_eps_20
save sigmoid_100_50_not_normalized/inputAbsDom22.mat A_eps_22
save sigmoid_100_50_not_normalized/inputAbsDom24.mat A_eps_24
save sigmoid_100_50_not_normalized/inputAbsDom26.mat A_eps_26

save sigmoid_100_50_not_normalized/inputRStar02.mat RS_eps_02
save sigmoid_100_50_not_normalized/inputRStar04.mat RS_eps_04
save sigmoid_100_50_not_normalized/inputRStar06.mat RS_eps_06
save sigmoid_100_50_not_normalized/inputRStar08.mat RS_eps_08
save sigmoid_100_50_not_normalized/inputRStar10.mat RS_eps_10
save sigmoid_100_50_not_normalized/inputRStar12.mat RS_eps_12
save sigmoid_100_50_not_normalized/inputRStar14.mat RS_eps_14
save sigmoid_100_50_not_normalized/inputRStar16.mat RS_eps_16
save sigmoid_100_50_not_normalized/inputRStar18.mat RS_eps_18
save sigmoid_100_50_not_normalized/inputRStar20.mat RS_eps_20
save sigmoid_100_50_not_normalized/inputRStar22.mat RS_eps_22
save sigmoid_100_50_not_normalized/inputRStar24.mat RS_eps_24
save sigmoid_100_50_not_normalized/inputRStar26.mat RS_eps_26

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

% save not_normalized/inputStar005.mat S_eps_005
% save not_normalized/inputStar010.mat S_eps_010 
% save not_normalized/inputStar015.mat S_eps_015  
% save not_normalized/inputStar020.mat S_eps_020 
% save not_normalized/inputStar025.mat S_eps_025 
% save not_normalized/inputStar030.mat S_eps_030
% save not_normalized/inputAbsDom005.mat A_eps_005 
% save not_normalized/inputAbsDom010.mat A_eps_010 
% save not_normalized/inputAbsDom015.mat A_eps_015 
% save not_normalized/inputAbsDom020.mat A_eps_020  
% save not_normalized/inputAbsDom025.mat A_eps_025 
% save not_normalized/inputAbsDom030.mat A_eps_030
% save not_normalized/inputRStar005.mat RS_eps_005 
% save not_normalized/inputRStar010.mat RS_eps_010 
% save not_normalized/inputRStar015.mat RS_eps_015 
% save not_normalized/inputRStar020.mat RS_eps_020 
% save not_normalized/inputRStar025.mat RS_eps_025 
% save not_normalized/inputRStar030.mat RS_eps_030