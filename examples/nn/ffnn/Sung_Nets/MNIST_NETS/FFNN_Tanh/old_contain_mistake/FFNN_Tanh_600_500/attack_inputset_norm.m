load images_normalized.mat;
load MNIST_tanh_600_500_normalized_DenseNet.mat;
N = size(IM_labels, 1); % number of test images used to testing robustness

figure                                          % initialize figure
colormap(gray)                                  % set to grayscale
for i = 1:50                                    % preview first 36 samples
    subplot(5,10,i)                              % plot them in 6 x 6 grid
    digit = reshape(IM_data(:,i), [28,28]);     % row = 28 x 28 image
    imagesc(digit)                              % show the image
    title(IM_labels(i))                   % show the label
end

% image perturbation by infinity norm attack (offsetting aimages)
eps = 0.005;
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_005(n) = Star(lb, ub);
   A_eps_norm_005(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_005(n) = RStar(lb, ub, inf);
end

eps = 0.010;
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_010(n) = Star(lb, ub);
   A_eps_norm_010(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_010(n) = RStar(lb, ub, inf);
end

eps = 0.015;
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_015(n) = Star(lb, ub);
   A_eps_norm_015(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_015(n) = RStar(lb, ub, inf);
end

eps = 0.020;
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_020(n) = Star(lb, ub);
   A_eps_norm_020(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_020(n) = RStar(lb, ub, inf);
end

eps = 0.025;
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_025(n) = Star(lb, ub);
   A_eps_norm_025(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_025(n) = RStar(lb, ub, inf);
end

eps = 0.030;
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_030(n) = Star(lb, ub);
   A_eps_norm_030(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_030(n) = RStar(lb, ub, inf);
end

save inputStar_norm005.mat S_eps_norm_005 
save inputStar_norm010.mat S_eps_norm_010 
save inputStar_norm015.mat S_eps_norm_015 
save inputStar_norm020.mat S_eps_norm_020 
save inputStar_norm025.mat S_eps_norm_025 
save inputStar_norm030.mat S_eps_norm_030
save inputAbsDom_norm005.mat A_eps_norm_005 
save inputAbsDom_norm010.mat A_eps_norm_010 
save inputAbsDom_norm015.mat A_eps_norm_015 
save inputAbsDom_norm020.mat A_eps_norm_020 
save inputAbsDom_norm025.mat A_eps_norm_025 
save inputAbsDom_norm030.mat A_eps_norm_030
save inputRStar_norm005.mat RS_eps_norm_005 
save inputRStar_norm010.mat RS_eps_norm_010 
save inputRStar_norm015.mat RS_eps_norm_015 
save inputRStar_norm020.mat RS_eps_norm_020
save inputRStar_norm025.mat RS_eps_norm_025 
save inputRStar_norm030.mat RS_eps_norm_030
