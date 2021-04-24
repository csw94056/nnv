load tanh_100_50_images_normalized.mat;
load MNIST_tanh_100_50_normalized_DenseNet.mat;
N = size(IM_labels, 1); % number of test images used to testing robustness

% figure                                          % initialize figure
% colormap(gray)                                  % set to grayscale
% for i = 1:50                                    % preview first 36 samples
%     subplot(5,10,i)                              % plot them in 6 x 6 grid
%     digit = reshape(IM_data(:,i), [28,28]);     % row = 28 x 28 image
%     imagesc(digit)                              % show the image
%     title(IM_labels(i))                   % show the label
% end

% image perturbation by infinity norm attack (offsetting aimages)
eps = 0.001
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0001(n) = Star(lb, ub);
   A_eps_norm_0001(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0001(n) = RStar(lb, ub, inf);
end

eps = 0.002
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0002(n) = Star(lb, ub);
   A_eps_norm_0002(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0002(n) = RStar(lb, ub, inf);
end

eps = 0.003
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0003(n) = Star(lb, ub);
   A_eps_norm_0003(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0003(n) = RStar(lb, ub, inf);
end

eps = 0.004
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0004(n) = Star(lb, ub);
   A_eps_norm_0004(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0004(n) = RStar(lb, ub, inf);
end

eps = 0.005
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0005(n) = Star(lb, ub);
   A_eps_norm_0005(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0005(n) = RStar(lb, ub, inf);
end

eps = 0.006
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0006(n) = Star(lb, ub);
   A_eps_norm_0006(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0006(n) = RStar(lb, ub, inf);
end

eps = 0.007
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0007(n) = Star(lb, ub);
   A_eps_norm_0007(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0007(n) = RStar(lb, ub, inf);
end

eps = 0.008
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0008(n) = Star(lb, ub);
   A_eps_norm_0008(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0008(n) = RStar(lb, ub, inf);
end

eps = 0.009
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0009(n) = Star(lb, ub);
   A_eps_norm_0009(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0009(n) = RStar(lb, ub, inf);
end

eps = 0.010
for n = 1:N
   IM = IM_data(:, n);
   lb = IM - eps;
   ub = IM + eps;
   S_eps_norm_0010(n) = Star(lb, ub);
   A_eps_norm_0010(n) = AbsDom(lb, ub, inf);
   RS_eps_norm_0010(n) = RStar(lb, ub, inf);
end

save tanh_100_50_normalized/inputStar_norm0001.mat S_eps_norm_0001
save tanh_100_50_normalized/inputStar_norm0002.mat S_eps_norm_0002
save tanh_100_50_normalized/inputStar_norm0003.mat S_eps_norm_0003
save tanh_100_50_normalized/inputStar_norm0004.mat S_eps_norm_0004
save tanh_100_50_normalized/inputStar_norm0005.mat S_eps_norm_0005
save tanh_100_50_normalized/inputStar_norm0006.mat S_eps_norm_0006
save tanh_100_50_normalized/inputStar_norm0007.mat S_eps_norm_0007
save tanh_100_50_normalized/inputStar_norm0008.mat S_eps_norm_0008
save tanh_100_50_normalized/inputStar_norm0009.mat S_eps_norm_0009
save tanh_100_50_normalized/inputStar_norm0010.mat S_eps_norm_0010

save tanh_100_50_normalized/inputAbsDom_norm0001.mat A_eps_norm_0001
save tanh_100_50_normalized/inputAbsDom_norm0002.mat A_eps_norm_0002
save tanh_100_50_normalized/inputAbsDom_norm0003.mat A_eps_norm_0003
save tanh_100_50_normalized/inputAbsDom_norm0004.mat A_eps_norm_0004
save tanh_100_50_normalized/inputAbsDom_norm0005.mat A_eps_norm_0005
save tanh_100_50_normalized/inputAbsDom_norm0006.mat A_eps_norm_0006
save tanh_100_50_normalized/inputAbsDom_norm0007.mat A_eps_norm_0007
save tanh_100_50_normalized/inputAbsDom_norm0008.mat A_eps_norm_0008
save tanh_100_50_normalized/inputAbsDom_norm0009.mat A_eps_norm_0009
save tanh_100_50_normalized/inputAbsDom_norm0010.mat A_eps_norm_0010

save tanh_100_50_normalized/inputRStar_norm0001.mat RS_eps_norm_0001 
save tanh_100_50_normalized/inputRStar_norm0002.mat RS_eps_norm_0002
save tanh_100_50_normalized/inputRStar_norm0003.mat RS_eps_norm_0003
save tanh_100_50_normalized/inputRStar_norm0004.mat RS_eps_norm_0004
save tanh_100_50_normalized/inputRStar_norm0005.mat RS_eps_norm_0005
save tanh_100_50_normalized/inputRStar_norm0006.mat RS_eps_norm_0006
save tanh_100_50_normalized/inputRStar_norm0007.mat RS_eps_norm_0007
save tanh_100_50_normalized/inputRStar_norm0008.mat RS_eps_norm_0008
save tanh_100_50_normalized/inputRStar_norm0009.mat RS_eps_norm_0009
save tanh_100_50_normalized/inputRStar_norm0010.mat RS_eps_norm_0010 
